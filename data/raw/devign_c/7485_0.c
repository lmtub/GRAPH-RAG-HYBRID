BlockErrorAction bdrv_get_error_action(BlockDriverState *bs, bool is_read, int error)

{

    BlockdevOnError on_err = is_read ? bs->on_read_error : bs->on_write_error;



    switch (on_err) {

    case BLOCKDEV_ON_ERROR_ENOSPC:

        return (error == ENOSPC) ?

               BLOCK_ERROR_ACTION_STOP : BLOCK_ERROR_ACTION_REPORT;

    case BLOCKDEV_ON_ERROR_STOP:

        return BLOCK_ERROR_ACTION_STOP;

    case BLOCKDEV_ON_ERROR_REPORT:

        return BLOCK_ERROR_ACTION_REPORT;

    case BLOCKDEV_ON_ERROR_IGNORE:

        return BLOCK_ERROR_ACTION_IGNORE;

    default:

        abort();

    }

}
