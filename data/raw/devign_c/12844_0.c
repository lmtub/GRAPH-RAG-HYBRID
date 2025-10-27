static inline int is_sector_in_chunk(BDRVDMGState* s,

		uint32_t chunk_num,int sector_num)

{

    if(chunk_num>=s->n_chunks || s->sectors[chunk_num]>sector_num ||

	    s->sectors[chunk_num]+s->sectorcounts[chunk_num]<=sector_num)

	return 0;

    else

	return -1;

}
