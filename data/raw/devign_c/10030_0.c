static void bdrv_sync_complete(void *opaque, int ret)

{

    /* do nothing. Masters do not directly interact with the backing store,

     * only the working copy so no mutexing required.

     */

}
