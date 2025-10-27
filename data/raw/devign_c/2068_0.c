BlockDriverState *bdrv_lookup_bs(const char *device,

                                 const char *node_name,

                                 Error **errp)

{

    BlockBackend *blk;

    BlockDriverState *bs;



    if (device) {

        blk = blk_by_name(device);



        if (blk) {

            return blk_bs(blk);

        }

    }



    if (node_name) {

        bs = bdrv_find_node(node_name);



        if (bs) {

            return bs;

        }

    }



    error_setg(errp, "Cannot find device=%s nor node_name=%s",

                     device ? device : "",

                     node_name ? node_name : "");

    return NULL;

}
