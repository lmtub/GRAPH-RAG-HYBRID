static int set_dirty_tracking(void)

{

    BlkMigDevState *bmds;

    int ret;



    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        bmds->dirty_bitmap = bdrv_create_dirty_bitmap(bmds->bs, BLOCK_SIZE,

                                                      NULL);

        if (!bmds->dirty_bitmap) {

            ret = -errno;

            goto fail;

        }

    }

    return 0;



fail:

    QSIMPLEQ_FOREACH(bmds, &block_mig_state.bmds_list, entry) {

        if (bmds->dirty_bitmap) {

            bdrv_release_dirty_bitmap(bmds->bs, bmds->dirty_bitmap);

        }

    }

    return ret;

}
