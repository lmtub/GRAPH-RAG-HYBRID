static void mirror_read_complete(void *opaque, int ret)

{

    MirrorOp *op = opaque;

    MirrorBlockJob *s = op->s;

    if (ret < 0) {

        BlockDriverState *source = s->common.bs;

        BlockErrorAction action;



        bdrv_set_dirty(source, op->sector_num, op->nb_sectors);

        action = mirror_error_action(s, true, -ret);

        if (action == BLOCK_ERROR_ACTION_REPORT && s->ret >= 0) {

            s->ret = ret;

        }



        mirror_iteration_done(op, ret);

        return;

    }

    bdrv_aio_writev(s->target, op->sector_num, &op->qiov, op->nb_sectors,

                    mirror_write_complete, op);

}
