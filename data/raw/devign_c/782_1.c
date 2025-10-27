static void vvfat_close(BlockDriverState *bs)

{

    BDRVVVFATState *s = bs->opaque;



    vvfat_close_current_file(s);

    array_free(&(s->fat));

    array_free(&(s->directory));

    array_free(&(s->mapping));

    if(s->cluster_buffer)

        free(s->cluster_buffer);

}
