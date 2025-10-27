void blk_apply_root_state(BlockBackend *blk, BlockDriverState *bs)

{

    bs->detect_zeroes = blk->root_state.detect_zeroes;

}
