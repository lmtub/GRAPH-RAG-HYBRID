static void encrypted_bdrv_it(void *opaque, BlockDriverState *bs)

{

    Error **errp = opaque;



    if (!error_is_set(errp) && bdrv_key_required(bs)) {

        error_set(errp, QERR_DEVICE_ENCRYPTED, bdrv_get_device_name(bs),

                  bdrv_get_encrypted_filename(bs));

    }

}
