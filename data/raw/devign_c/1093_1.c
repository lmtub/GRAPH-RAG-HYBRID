static uint64_t qemu_rdma_make_wrid(uint64_t wr_id, uint64_t index,

                                         uint64_t chunk)

{

    uint64_t result = wr_id & RDMA_WRID_TYPE_MASK;



    result |= (index << RDMA_WRID_BLOCK_SHIFT);

    result |= (chunk << RDMA_WRID_CHUNK_SHIFT);



    return result;

}
