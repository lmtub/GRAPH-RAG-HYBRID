static void dmg_close(BlockDriverState *bs)

{

    BDRVDMGState *s = bs->opaque;

    close(s->fd);

    if(s->n_chunks>0) {

	free(s->types);

	free(s->offsets);

	free(s->lengths);

	free(s->sectors);

	free(s->sectorcounts);

    }

    free(s->compressed_chunk);

    free(s->uncompressed_chunk);

    inflateEnd(&s->zstream);

}
