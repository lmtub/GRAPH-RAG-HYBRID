static void data_plane_set_up_op_blockers(VirtIOBlockDataPlane *s)

{

    assert(!s->blocker);

    error_setg(&s->blocker, "block device is in use by data plane");

    blk_op_block_all(s->conf->conf.blk, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_RESIZE, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_DRIVE_DEL, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_BACKUP_SOURCE, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_CHANGE, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_COMMIT_SOURCE, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_COMMIT_TARGET, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_EJECT, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_EXTERNAL_SNAPSHOT,

                   s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_INTERNAL_SNAPSHOT,

                   s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_INTERNAL_SNAPSHOT_DELETE,

                   s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_MIRROR_SOURCE, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_STREAM, s->blocker);

    blk_op_unblock(s->conf->conf.blk, BLOCK_OP_TYPE_REPLACE, s->blocker);

}
