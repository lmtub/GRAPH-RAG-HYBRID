static void block_set_params(const MigrationParams *params, void *opaque)

{

    block_mig_state.blk_enable = params->blk;

    block_mig_state.shared_base = params->shared;



    /* shared base means that blk_enable = 1 */

    block_mig_state.blk_enable |= params->shared;

}
