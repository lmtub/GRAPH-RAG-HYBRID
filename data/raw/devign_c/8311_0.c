static int context_init(H264Context *h){

    CHECKED_ALLOCZ(h->top_borders[0], h->s.mb_width * (16+8+8) * sizeof(uint8_t))

    CHECKED_ALLOCZ(h->top_borders[1], h->s.mb_width * (16+8+8) * sizeof(uint8_t))



    return 0;

fail:

    return -1; // free_tables will clean up for us

}
