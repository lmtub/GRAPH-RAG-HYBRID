int coroutine_fn bdrv_co_flush(BlockDriverState *bs)

{

    int ret;



    if (!bs || !bdrv_is_inserted(bs) || bdrv_is_read_only(bs)) {

        return 0;

    }



    /* Write back cached data to the OS even with cache=unsafe */

    BLKDBG_EVENT(bs->file, BLKDBG_FLUSH_TO_OS);

    if (bs->drv->bdrv_co_flush_to_os) {

        ret = bs->drv->bdrv_co_flush_to_os(bs);

        if (ret < 0) {

            return ret;

        }

    }



    /* But don't actually force it to the disk with cache=unsafe */

    if (bs->open_flags & BDRV_O_NO_FLUSH) {

        goto flush_parent;

    }



    BLKDBG_EVENT(bs->file, BLKDBG_FLUSH_TO_DISK);

    if (bs->drv->bdrv_co_flush_to_disk) {

        ret = bs->drv->bdrv_co_flush_to_disk(bs);

    } else if (bs->drv->bdrv_aio_flush) {

        BlockAIOCB *acb;

        CoroutineIOCompletion co = {

            .coroutine = qemu_coroutine_self(),

        };



        acb = bs->drv->bdrv_aio_flush(bs, bdrv_co_io_em_complete, &co);

        if (acb == NULL) {

            ret = -EIO;

        } else {

            qemu_coroutine_yield();

            ret = co.ret;

        }

    } else {

        /*

         * Some block drivers always operate in either writethrough or unsafe

         * mode and don't support bdrv_flush therefore. Usually qemu doesn't

         * know how the server works (because the behaviour is hardcoded or

         * depends on server-side configuration), so we can't ensure that

         * everything is safe on disk. Returning an error doesn't work because

         * that would break guests even if the server operates in writethrough

         * mode.

         *

         * Let's hope the user knows what he's doing.

         */

        ret = 0;

    }

    if (ret < 0) {

        return ret;

    }



    /* Now flush the underlying protocol.  It will also have BDRV_O_NO_FLUSH

     * in the case of cache=unsafe, so there are no useless flushes.

     */

flush_parent:

    return bdrv_co_flush(bs->file);

}
