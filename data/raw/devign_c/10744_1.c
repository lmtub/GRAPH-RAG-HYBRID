static inline uint8_t *ram_chunk_start(const RDMALocalBlock *rdma_ram_block,

                                       uint64_t i)

{

    return (uint8_t *) (((uintptr_t) rdma_ram_block->local_host_addr)

                                    + (i << RDMA_REG_CHUNK_SHIFT));

}
