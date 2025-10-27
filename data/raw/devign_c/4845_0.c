static inline int dmg_read_chunk(BlockDriverState *bs, int sector_num)

{

    BDRVDMGState *s = bs->opaque;



    if(!is_sector_in_chunk(s,s->current_chunk,sector_num)) {

	int ret;

	uint32_t chunk = search_chunk(s,sector_num);



	if(chunk>=s->n_chunks)

	    return -1;



	s->current_chunk = s->n_chunks;

	switch(s->types[chunk]) {

	case 0x80000005: { /* zlib compressed */

	    int i;



	    /* we need to buffer, because only the chunk as whole can be

	     * inflated. */

	    i=0;

	    do {

                ret = bdrv_pread(bs->file, s->offsets[chunk] + i,

                                 s->compressed_chunk+i, s->lengths[chunk]-i);

		if(ret<0 && errno==EINTR)

		    ret=0;

		i+=ret;

	    } while(ret>=0 && ret+i<s->lengths[chunk]);



	    if (ret != s->lengths[chunk])

		return -1;



	    s->zstream.next_in = s->compressed_chunk;

	    s->zstream.avail_in = s->lengths[chunk];

	    s->zstream.next_out = s->uncompressed_chunk;

	    s->zstream.avail_out = 512*s->sectorcounts[chunk];

	    ret = inflateReset(&s->zstream);

	    if(ret != Z_OK)

		return -1;

	    ret = inflate(&s->zstream, Z_FINISH);

	    if(ret != Z_STREAM_END || s->zstream.total_out != 512*s->sectorcounts[chunk])

		return -1;

	    break; }

	case 1: /* copy */

	    ret = bdrv_pread(bs->file, s->offsets[chunk],

                             s->uncompressed_chunk, s->lengths[chunk]);

	    if (ret != s->lengths[chunk])

		return -1;

	    break;

	case 2: /* zero */

	    memset(s->uncompressed_chunk, 0, 512*s->sectorcounts[chunk]);

	    break;

	}

	s->current_chunk = chunk;

    }

    return 0;

}
