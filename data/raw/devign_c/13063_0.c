static int dmg_read(BlockDriverState *bs, int64_t sector_num,

                    uint8_t *buf, int nb_sectors)

{

    BDRVDMGState *s = bs->opaque;

    int i;



    for(i=0;i<nb_sectors;i++) {

	uint32_t sector_offset_in_chunk;

	if(dmg_read_chunk(bs, sector_num+i) != 0)

	    return -1;

	sector_offset_in_chunk = sector_num+i-s->sectors[s->current_chunk];

	memcpy(buf+i*512,s->uncompressed_chunk+sector_offset_in_chunk*512,512);

    }

    return 0;

}
