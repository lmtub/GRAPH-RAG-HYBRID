static int dmg_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVDMGState *s = bs->opaque;

    off_t info_begin,info_end,last_in_offset,last_out_offset;

    uint32_t count;

    uint32_t max_compressed_size=1,max_sectors_per_chunk=1,i;

    int64_t offset;



    s->fd = open(filename, O_RDONLY | O_BINARY);

    if (s->fd < 0)

        return -errno;

    bs->read_only = 1;

    s->n_chunks = 0;

    s->offsets = s->lengths = s->sectors = s->sectorcounts = NULL;



    /* read offset of info blocks */

    offset = lseek(s->fd, -0x1d8, SEEK_END);

    if (offset < 0) {

        goto fail;

    }



    info_begin = read_off(s->fd, offset);

    if (info_begin == 0) {

	goto fail;

    }



    if (read_uint32(s->fd, info_begin) != 0x100) {

        goto fail;

    }



    count = read_uint32(s->fd, info_begin + 4);

    if (count == 0) {

        goto fail;

    }

    info_end = info_begin + count;



    offset = info_begin + 0x100;



    /* read offsets */

    last_in_offset = last_out_offset = 0;

    while (offset < info_end) {

        uint32_t type;



	count = read_uint32(s->fd, offset);

	if(count==0)

	    goto fail;

        offset += 4;



	type = read_uint32(s->fd, offset);

	if (type == 0x6d697368 && count >= 244) {

	    int new_size, chunk_count;



            offset += 4;

            offset += 200;



	    chunk_count = (count-204)/40;

	    new_size = sizeof(uint64_t) * (s->n_chunks + chunk_count);

	    s->types = qemu_realloc(s->types, new_size/2);

	    s->offsets = qemu_realloc(s->offsets, new_size);

	    s->lengths = qemu_realloc(s->lengths, new_size);

	    s->sectors = qemu_realloc(s->sectors, new_size);

	    s->sectorcounts = qemu_realloc(s->sectorcounts, new_size);



	    for(i=s->n_chunks;i<s->n_chunks+chunk_count;i++) {

		s->types[i] = read_uint32(s->fd, offset);

		offset += 4;

		if(s->types[i]!=0x80000005 && s->types[i]!=1 && s->types[i]!=2) {

		    if(s->types[i]==0xffffffff) {

			last_in_offset = s->offsets[i-1]+s->lengths[i-1];

			last_out_offset = s->sectors[i-1]+s->sectorcounts[i-1];

		    }

		    chunk_count--;

		    i--;

		    offset += 36;

		    continue;

		}

		offset += 4;



		s->sectors[i] = last_out_offset+read_off(s->fd, offset);

		offset += 8;



		s->sectorcounts[i] = read_off(s->fd, offset);

		offset += 8;



		s->offsets[i] = last_in_offset+read_off(s->fd, offset);

		offset += 8;



		s->lengths[i] = read_off(s->fd, offset);

		offset += 8;



		if(s->lengths[i]>max_compressed_size)

		    max_compressed_size = s->lengths[i];

		if(s->sectorcounts[i]>max_sectors_per_chunk)

		    max_sectors_per_chunk = s->sectorcounts[i];

	    }

	    s->n_chunks+=chunk_count;

	}

    }



    /* initialize zlib engine */

    s->compressed_chunk = qemu_malloc(max_compressed_size+1);

    s->uncompressed_chunk = qemu_malloc(512*max_sectors_per_chunk);

    if(inflateInit(&s->zstream) != Z_OK)

	goto fail;



    s->current_chunk = s->n_chunks;



    return 0;

fail:

    close(s->fd);

    return -1;

}
