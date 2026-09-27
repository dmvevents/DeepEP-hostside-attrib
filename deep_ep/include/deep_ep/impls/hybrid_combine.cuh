#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/combine_utils.cuh>

namespace deep_ep::elastic {

template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs,
          int kNumScaleupWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),
          int kNumScaleupUpdateInterval = 3,
          int kNumChannelsPerSM = kNumForwardWarps,
          int kNumChannels = kNumChannelsPerSM * kNumSMs,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumWarps = kNumScaleupWarps + kNumForwardWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          bool kUseScaleoutRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          bool kUseScaleupRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>(),
          int kNumTokensInScaleoutLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          int kNumTokensInScaleupLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>()>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_combine_impl(nv_bfloat16* x,
                    float* topk_weights,
                    int* src_metadata,
                    int* psum_num_recv_tokens_per_scaleup_rank,
                    int* token_metadata_at_forward,
                    int* channel_linked_list,
                    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                    void* buffer, void* workspace,
                    const int scaleout_rank_idx, const int scaleup_rank_idx,
                    int num_reduced_tokens) {
    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;

    // Combine vector type selection
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

    // We should assign the real number of received tokens if without CPU sync
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumScaleupRanks - 1);

    // Token layouts
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);

    // TMA buffers
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto tma_buffer = layout::BufferLayout<true>(
        token_layout, kNumWarps, 1, smem).get_rank_buffer(warp_idx).get_token_buffer(0);

    // All the buffer layouts
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleupLayout, kNumScaleoutRanks * kNumMaxTokensPerRank,
        buffer);
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleoutLayout, kNumMaxTokensPerRank,
        scaleup_buffer.get_buffer_end_ptr());
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, kAllowMultipleReduction ? 1 : kNumTopk, kNumChannels * (kNumScaleoutRanks * kNumMaxTokensPerChannel),
        scaleout_recv_buffer.get_buffer_end_ptr());

    // Init TMA for scale-up and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // NCCL Gin handle
    // Each warp is a channel
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, warp_idx % kNumChannelsPerSM);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Global parallel barriers for scale-out subteam and scale-up subteam
    // NOTES: this barrier needs a grid sync, as there are channel scale-up tail cleaning before
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // Adjust register count at certain cases
    // TODO: support more cases, or try to make channel count more aligned
    const bool kAdjustRegisters = (kNumChannelsPerSM == 4 or kNumChannelsPerSM == 8) and not kUseExpandedLayout;
    constexpr int kNumRegistersForScaleupWarps = 40;
    constexpr int kNumRegistersForForwardWarps = 256 - kNumRegistersForScaleupWarps;

    // Different warp roles
    if (warp_idx < kNumScaleupWarps) {
        const auto channel_idx = sm_idx * kNumChannelsPerSM + warp_idx;

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_dealloc<kNumRegistersForScaleupWarps>();

        // Shift into the right buffer if using rank layout
        if constexpr (kUseScaleupRankLayout)
            scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Expanding send mode must not be backward
        if constexpr (kDoExpandedSend)
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // Tail issuer
        // `st.release.sys` is pretty slow, so do it by an interval
        int update_counter = 0;
        int stored_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        int stored_old_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
        const auto update_tails = [&](const bool& finish = false) {
            ++ update_counter;
            if (finish or update_counter == kNumScaleupUpdateInterval) {
                // Wait all TMA stores to finish
                ptx::tma_store_wait();
                __syncwarp();

                // Issue
                #pragma unroll
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                    if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                        // NOTES: save some traffic with `stored_old_num_tokens_sent`
                        // Also, we cannot rewrite a finished slot, if the peer is going to clean it
                        if (stored_num_tokens_sent[i] != stored_old_num_tokens_sent[i])
                            ptx::st_release_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                        stored_old_num_tokens_sent[i] = stored_num_tokens_sent[i];
                    }
                }
                update_counter = 0;
            }
            __syncwarp();
        };

        // Shape of `channel_linked_list`: `[kNumChannels, kNumMaxTokensPerChannel + 1, kNumScaleupRanks]`
        // Iterate until all scale-up peers finish
        int dst_scaleup_rank_idx = channel_idx;
        int stored_ll_idx[kNumScaleupRanksPerLane] = {}, stored_token_idx[kNumScaleupRanksPerLane] = {};
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
            stored_token_idx[i] = -1;
        while (true) {
            // Load token indices in the list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                const auto j = i * 32 + lane_idx;
                stored_token_idx[i] = i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks ?
                    __ldg(channel_linked_list +
                          channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumScaleupRanks +
                          stored_ll_idx[i] * kNumScaleupRanks + j) : -1;
            }
            __syncwarp();

            // Check whether all ranks are finished
            bool exited = true;
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                exited &= ptx::all(stored_token_idx[i] < 0);
            if (exited)
                break;

            // Process tokens for all ranks together using bitmask to skip inactive ranks
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up ranks for 64-bit mask");
            using mask_t = std::conditional_t<(kNumScaleupRanks <= 32), uint32_t, uint64_t>;
            mask_t wip_mask = 0;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                wip_mask |= static_cast<mask_t>(ptx::gather(stored_token_idx[j] >= 0)) << (j * 32);
            while (wip_mask) {
                // Find next active rank after `dst_scaleup_rank_idx` (round-robin)
                const auto start = (dst_scaleup_rank_idx + 1) % kNumScaleupRanks;
                const auto hi_mask = (wip_mask >> start) << start;
                dst_scaleup_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);
                wip_mask ^= static_cast<mask_t>(1) << dst_scaleup_rank_idx;

                // Exchange token index from the owning lane using static partition iteration
                int token_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = dst_scaleup_rank_idx - j * 32;
                    token_idx = src_lane_idx == lane_idx ? stored_token_idx[j] : token_idx;
                }
                token_idx = ptx::exchange(token_idx, dst_scaleup_rank_idx % 32);

                // Get source metadata and decide the destination buffer
                constexpr int kMetadataStride = 2 + kNumTopk;
                const auto src_global_token_idx = __ldg(src_metadata + token_idx * kMetadataStride + 0);
                const auto src_token_idx = src_global_token_idx % kNumMaxTokensPerRank;
                const auto src_scaleout_rank_idx = src_global_token_idx / (kNumMaxTokensPerRank * kNumScaleupRanks);
                auto token_buffer = [&]() {
                    if constexpr (kUseScaleupRankLayout) {
                        const auto src_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) / kNumTopk;
                        return scaleup_buffer.get_token_buffer(src_slot_idx);
                    } else {
                        const auto master_topk_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) % kNumTopk;
                        return scaleup_buffer
                            .get_rank_buffer(master_topk_idx)
                            .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                    }
                }();
                token_buffer.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(token_buffer.get_base_ptr(), dst_scaleup_rank_idx));

                // Some checks
                EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");

                // Read source indices for expand mode
                int stored_topk_slot_idx = -1;
                if constexpr (kUseExpandedLayout) {
                    if (lane_idx < kNumTopk)
                        stored_topk_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + (2 + lane_idx));
                    __syncwarp();
                }

                // 3 cases:
                //  - no-expand, expand + no-reduce
                //  - expand + reduce
                //  - expand + send all
                auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                if (no_local_reduce) {
                    int token_idx_in_tensor = token_idx;
                    if constexpr (kUseExpandedLayout)
                        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

                    // Directly load
                    if (ptx::elect_one_sync()) {
                        const auto load_ptr =
                            math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                        ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    __syncwarp();
                } else if constexpr (kAllowMultipleReduction) {
                    // Do local reduction
                    // Sort valid top-k indices to front
                    int topk_slot_idx[kNumTopk];
                    compute_topk_slots(
                        topk_slot_idx, reduce_valid_mask,
                        [=](const int& idx) {
                            return ptx::exchange(stored_topk_slot_idx, idx);
                        }
                    );

                    // Reduce into shared memory
                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
                    combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                        /* Get source base */ [=](const int& slot_idx) {
                            return math::advance_ptr<combine_vec_t>(
                                x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                        },
                        /* Wait buffer release */ [=]() {
                            ptx::tma_store_wait();
                            __syncwarp();
                        }
                    );
                    ptx::tma_store_fence();
                    __syncwarp();
                } else {
                    // No local reduction, send all data (expanded send)
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        int topk_slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                        if (topk_slot_idx < 0)
                            continue;

                        if (ptx::elect_one_sync()) {
                            // Load
                            const auto load_ptr = math::advance_ptr(x, static_cast<int64_t>(kDoExpandedSend ? topk_slot_idx : token_idx) * kNumHiddenBytes);
                            ptx::tma_store_wait();
                            ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            // NOTES: We don't need to care about `topk_weights` since we are in expand mode

                            // Store
                            const auto dst_token_buffer = scaleup_buffer
                                .get_rank_buffer(k)
                                .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                            ptx::tma_store_1d(
                                gin.get_sym_ptr<ncclTeamTagLsa>(dst_token_buffer.get_base_ptr(), dst_scaleup_rank_idx),
                                tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                            ptx::tma_store_commit();
                        }
                        __syncwarp();
                    }
                }

                // Write top-k weights (expanded send handled inside the loop above)
                if (not kDoExpandedSend and topk_weights != nullptr and lane_idx < kNumTopk) {
                    float value = 0;
                    if constexpr (kUseExpandedLayout) {
                        if (stored_topk_slot_idx >= 0)
                            value = __ldg(topk_weights + stored_topk_slot_idx);
                    } else {
                        value = __ldg(topk_weights + (token_idx * kNumTopk + lane_idx));
                    }
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = value;
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Issue TMA stores into remote scale-up buffer
                // NOTES: `kDoExpandedSend` mode has already issued
                if (not kDoExpandedSend and ptx::elect_one_sync()) {
                    // Wait TMA arrival (only for non-reduced cases)
                    if (no_local_reduce) {
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }

                    // Issue stores
                    ptx::tma_store_1d(
                        token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                        token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                __syncwarp();
            }

            // Update the tails together
            // NOTES: TMA wait is inside
            update_tails();

            // Move linked list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                stored_ll_idx[i] += (stored_token_idx[i] >= 0);
        }

        // Update for the unissued ones
        update_tails(true);
    } else {
        const auto forward_warp_idx = warp_idx - kNumScaleupWarps;
        const auto channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_alloc<kNumRegistersForForwardWarps>();

        // Shift into the right buffer
        scaleout_send_buffer = scaleout_send_buffer.get_channel_buffer<kNumScaleoutRanks * kNumMaxTokensPerChannel>(channel_idx);

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // NH-A2: count-gated completion with ONE terminal STRONG signal per (channel, remote peer) instead of NH-A's
        // per-put SignalAdd (the deprecated `ncclGin_SignalAdd` is WEAK unless `ginStrongSignalsRequired`, which this tree
        // never set: gin_host.cc:255). `ncclGin_StrongSignalAdd`: "visibility implies all preceding puts are settled"
        // (nccl_device/gin.h:59; put(): "this put AND all preceding puts on this context to the same peer"). The peer's
        // return puts for this channel travel on this warp's context (`get_qp_mode` is rank-independent) and its terminal
        // signal follows them on the same context, so `readSignal >= target` gates the data with no sender-side flush and
        // no dispatch-side expected-count tables (NH-A's per-buffer table and its G3 scope limit are gone). The signal-only
        // op is non-aggregate (default flags): on a device-posting backend it also rings a doorbell deferred by
        // AggregateRequests puts; on the proxy backend the device flag is never transported (gin_proxy.h buildGfd).
        EP_STATIC_ASSERT((comm::constexpr_max_qp_channel_slots<kNumSMs, kNumQPs, kNumChannelsPerSM>() <=
                          layout::WorkspaceLayout::kNumNHACombineSignalSlots), "NH-A2: channel/QP shape exceeds the indexed-signal budget");
        const auto nha_signal_id = static_cast<ncclGinSignal_t>(
            kNumRanks + 4 + comm::get_qp_channel_slot<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, forward_warp_idx));
        // Local-completion point for `scaleout_send_buffer` reuse stays AFTER the wait (hazard i; G4 reuse probe).
        constexpr bool kNHAFlushAfterWait = true;
        // Every remote scale-out peer sends exactly one terminal signal per channel per launch, unconditionally (as the
        // baseline's per-peer `red_add_rel` did), so the expected count is a compile-time constant.
        constexpr int nha_expected = kNumScaleoutRanks - 1;

        // Overlap TMA stores and reduction
        int last_src_scaleout_rank_idx = -1;
        int last_is_token_last_in_chunk = 0;
        void* last_recv_token_buffer_ptr = nullptr;
        void* last_send_token_buffer_ptr = nullptr;
        const auto flush_last_tma_and_issue_rdma = [&]() {
            if (last_src_scaleout_rank_idx >= 0 and ptx::elect_one_sync()) {
                ptx::tma_store_wait();

                // Issue only if not local rank
                if (last_src_scaleout_rank_idx != scaleout_rank_idx) {
                    gin.put<ncclTeamTagRail>(
                        last_recv_token_buffer_ptr,
                        last_send_token_buffer_ptr,
                        token_layout.get_num_bytes<false>(),
                        last_src_scaleout_rank_idx,
                        last_is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests
                    );
                }
            }
            __syncwarp();
        };

        // Replay the dispatch
        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};
        for (int i = 0; ; ++ i) {
            const auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            const auto is_token_last_in_chunk = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 1);
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;
            const auto src_scaleout_rank_idx = src_rank_idx / kNumScaleupRanks;
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + lane_idx) : -1;
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + kNumTopk + lane_idx) : -1;
            if (src_token_global_idx < 0)
                break;

            // Scaleup rank mask
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up peers");
            using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
            const auto scaleup_mask = ptx::reduce_or(
                stored_src_scaleup_rank_idx >= 0 ?
                (mask_t(1) << stored_src_scaleup_rank_idx) : mask_t(0));
            bool stored_is_scaleup_rank_needed[kNumScaleupRanksPerLane];
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_is_scaleup_rank_needed[j] = (scaleup_mask >> (j * 32 + lane_idx)) & 1;

            // Wait all tails to arrive
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                bool arrived = true;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    arrived &= not stored_is_scaleup_rank_needed[j] or stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j];
                if (ptx::all(arrived))
                    return true;

                // Reload cached
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto k = j * 32 + lane_idx;
                    stored_cached_scaleup_tail[j] = j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks ?
                        ptx::ld_acquire_sys(workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k)) : -1;
                }

                // Timeout
                if (is_last_check) {
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        printf("DeepEP combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                               channel_idx, j * 32 + lane_idx,
                               stored_num_tokens_recv[j],
                               stored_cached_scaleup_tail[j],
                               stored_is_scaleup_rank_needed[j]);
                    }
                }
                return false;
            });

            // Increase received count
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);
            
            if constexpr (not kAllowMultipleReduction) {
                // Cases where multiple reduction is disabled. We need to forward all data from scaleup peers to scaleout peers
                // TODO: Let scale-up warps directly put data into `send_buffer`?
                const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                auto topk_valid_mask = kUseExpandedLayout ?
                    ptx::gather(stored_src_scaleup_rank_idx >= 0) :
                    ptx::gather(ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);  // Deduplicate w.r.t. scaleup rank index if expanded mode is disabled
                if (ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        if ((topk_valid_mask & (1u << k)) == 0u)
                            continue;

                        // Issue TMA load, and wait
                        ptx::tma_load_1d(
                            tma_buffer.get_base_ptr(), scaleup_buffer.get_rank_buffer(k).get_token_buffer(src_slot_idx).get_base_ptr(),
                            mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        // Issue TMA store, and wait
                        const auto recv_buffer_ptr = scaleout_recv_buffer.get_rank_buffer(k).get_token_buffer(src_token_idx).get_base_ptr();
                        const auto send_buffer_ptr = src_scaleout_rank_idx == scaleout_rank_idx ?
                            recv_buffer_ptr : scaleout_send_buffer.get_rank_buffer(k).get_token_buffer(i).get_base_ptr();
                        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                        ptx::tma_store_commit();
                        ptx::tma_store_wait();

                        // Issue IBGDA
                        topk_valid_mask ^= 1u << k;
                        if (src_scaleout_rank_idx != scaleout_rank_idx) {
                            gin.put<ncclTeamTagRail>(
                                recv_buffer_ptr,
                                send_buffer_ptr,
                                token_layout.get_num_bytes<false>(),
                                src_scaleout_rank_idx,
                                topk_valid_mask == 0 and is_token_last_in_chunk ? 0 : ncclGinOptFlagsAggregateRequests
                            );
                        }
                    }
                }
                __syncwarp();
            } else {
                // NOTES: we must do deduplicate and only add once from one rank
                auto reduce_valid_mask = ptx::gather(
                    ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);

                // Calculate the source buffer index
                int stored_src_buffer_idx = 0;
                if constexpr (kUseScaleupRankLayout) {
                    stored_src_buffer_idx =
                        stored_src_scaleup_rank_idx * scaleup_buffer.num_max_tokens_per_rank + stored_src_slot_idx;
                } else {
                    const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                    stored_src_buffer_idx = stored_src_slot_idx == -1 ? -1 :
                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;
                }
                
                // Preprocess top-k indices
                int topk_slot_idx[kNumTokensInScaleupLayout];
                compute_topk_slots(
                    topk_slot_idx, reduce_valid_mask,
                    [=](const int& idx) {
                        return ptx::exchange(stored_src_buffer_idx, idx);
                    }
                );

                // Do reduce
                constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, kAdjustRegisters ? 8 : 4>();
                combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumScaleoutRanks)>(
                    lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                    /* Get source base */ [=](const int& slot_idx) {
                        return static_cast<combine_vec_t*>(scaleup_buffer.get_token_buffer(slot_idx, true).get_base_ptr());
                    },
                    /* Wait buffer release */ [=]() {
                        flush_last_tma_and_issue_rdma();
                    }
                );

                // Merge topk weights
                // NOTES: the slot indices must follow the master lane
                stored_src_buffer_idx = ptx::exchange(
                    stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                if (stored_src_scaleup_rank_idx >= 0) {
                    tma_buffer.get_topk_weights_ptr()[lane_idx] =
                        scaleup_buffer.get_token_buffer(stored_src_buffer_idx, true)
                                    .get_topk_weights_ptr()[lane_idx];
                }
                ptx::tma_store_fence();
                __syncwarp(); // Necessary to let the leader lane see the writes

                // Assign send and receive buffers
                // NOTES: as we only have 1 destination, we will use "send" as "recv" for local transfer
                int scaleout_recv_buffer_rank_idx;
                if constexpr (kUseScaleoutRankLayout) {
                    scaleout_recv_buffer_rank_idx = scaleout_rank_idx;
                } else {
                    const int src_topk_idx = ptx::get_master_lane_idx(ptx::gather(stored_src_scaleup_rank_idx >= 0));
                    scaleout_recv_buffer_rank_idx = src_topk_idx;
                }
                const auto recv_token_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_recv_buffer_rank_idx).get_token_buffer(src_token_idx);
                const auto send_token_buffer = src_scaleout_rank_idx == scaleout_rank_idx ?
                    recv_token_buffer :
                    scaleout_send_buffer.get_token_buffer(i);

                // Write into scale-out send buffer or local rank recv buffer bypass
                if (ptx::elect_one_sync()) {
                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                                    token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                __syncwarp();

                // Record RDMA info to issue later
                last_src_scaleout_rank_idx = src_scaleout_rank_idx;
                last_is_token_last_in_chunk = is_token_last_in_chunk;
                last_recv_token_buffer_ptr = recv_token_buffer.get_base_ptr();
                last_send_token_buffer_ptr = send_token_buffer.get_base_ptr();
            }
        }

        // Issue the last RDMA
        if constexpr (kAllowMultipleReduction)
            flush_last_tma_and_issue_rdma();

        // NH-A2: one terminal signal per remote scale-out peer on this channel's context, AFTER this warp's last put to it.
        // Strong-capable backend (the measured PROXY backend reports supportsStrongSignals = true, gin_host_proxy.cc:348): a
        // STRONG signal is visible at the peer only once all preceding puts on this context to that peer are settled
        // (gin.h:59) -- exactly the wait's data-completion contract, with no sender-side flush on the peers' critical path
        // (the baseline flushed here). Otherwise the API rule applies (gin_device_common.h:190-192): flush first, then a WEAK
        // signal carries the same guarantee -- so the fallback never asks a weak-only backend to lower a strong action.
        const bool nha_strong = gin.gin._supportsStrongSignal();   // warp-uniform: a per-context constant
        if (not nha_strong)
            gin.flush<ncclCoopWarp>();
        __syncwarp();
        if (lane_idx < kNumScaleoutRanks and lane_idx != scaleout_rank_idx) {
            if (nha_strong)
                gin.signal<ncclTeamTagRail>(lane_idx, ncclGin_StrongSignalAdd{nha_signal_id, static_cast<uint64_t>(1)});
            else
                gin.signal<ncclTeamTagRail>(lane_idx, ncclGin_WeakSignalAdd{nha_signal_id, static_cast<uint64_t>(1)});
        }
        __syncwarp();

        // Clean scaleup tails
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
            const auto k = j * 32 + lane_idx;
            if (j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks)
                *workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k) = 0;
        }
        __syncwarp();

        // NH-A2: count-gated wait (unchanged from NH-A). `target` is shadow-relative (the shadow is bumped by this launch's
        // constant expected count); the read is acquire-ordered on the GIN context the peers' puts + terminal signals use.
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid ranks");
        if (ptx::elect_one_sync()) {
            const auto shadow_ptr = gin.gin.getSignalShadowPtr(nha_signal_id);
            const auto target = (*shadow_ptr += static_cast<uint64_t>(nha_expected));
            comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = gin.gin.readSignal(nha_signal_id, 64, cuda::memory_order_acquire);
                if (signal >= target)
                    return true;

                if (is_last_check) {
                    printf("DeepEP combine (NH-A2 count-gated wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                           "channel: %d, signal id: %d, signal: %llu, target: %llu, expected this launch: %d\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                           channel_idx, static_cast<int>(nha_signal_id),
                           static_cast<unsigned long long>(signal), static_cast<unsigned long long>(target), nha_expected);
                }
                return false;
            });
        }
        __syncwarp();

        // Local completion of this warp's own puts (protects `scaleout_send_buffer` reuse, hazard i) -- now behind the
        // wait, so no peer's critical path depends on it.
        if constexpr (kNHAFlushAfterWait)
            gin.flush<ncclCoopWarp>();
        __syncwarp();
    }

    // No barrier at epilogue
}

}  // namespace deep_ep::elastic
