static int blk_root_inactivate(BdrvChild *child)

{

    BlockBackend *blk = child->opaque;



    if (blk->disable_perm) {

        return 0;

    }



    /* Only inactivate BlockBackends for guest devices (which are inactive at

     * this point because the VM is stopped) and unattached monitor-owned

     * BlockBackends. If there is still any other user like a block job, then

     * we simply can't inactivate the image. */

    if (!blk->dev && !blk->name[0]) {

        return -EPERM;

    }



    blk->disable_perm = true;

    if (blk->root) {

        bdrv_child_try_set_perm(blk->root, 0, BLK_PERM_ALL, &error_abort);

    }



    return 0;

}
