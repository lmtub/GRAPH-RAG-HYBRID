void bdrv_error_action(BlockDriverState *bs, BlockErrorAction action,

                       bool is_read, int error)

{

    assert(error >= 0);

    bdrv_emit_qmp_error_event(bs, QEVENT_BLOCK_IO_ERROR, action, is_read);

    if (action == BDRV_ACTION_STOP) {

        vm_stop(RUN_STATE_IO_ERROR);

        bdrv_iostatus_set_err(bs, error);

    }

}
