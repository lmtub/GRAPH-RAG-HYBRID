static void parallels_close(BlockDriverState *bs)

{

    BDRVParallelsState *s = bs->opaque;

    g_free(s->catalog_bitmap);

}
