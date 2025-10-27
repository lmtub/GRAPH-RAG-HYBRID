unsigned long hbitmap_iter_skip_words(HBitmapIter *hbi)

{

    size_t pos = hbi->pos;

    const HBitmap *hb = hbi->hb;

    unsigned i = HBITMAP_LEVELS - 1;



    unsigned long cur;

    do {

        cur = hbi->cur[--i];

        pos >>= BITS_PER_LEVEL;

    } while (cur == 0);



    /* Check for end of iteration.  We always use fewer than BITS_PER_LONG

     * bits in the level 0 bitmap; thus we can repurpose the most significant

     * bit as a sentinel.  The sentinel is set in hbitmap_alloc and ensures

     * that the above loop ends even without an explicit check on i.

     */



    if (i == 0 && cur == (1UL << (BITS_PER_LONG - 1))) {

        return 0;

    }

    for (; i < HBITMAP_LEVELS - 1; i++) {

        /* Shift back pos to the left, matching the right shifts above.

         * The index of this word's least significant set bit provides

         * the low-order bits.

         */

        pos = (pos << BITS_PER_LEVEL) + ffsl(cur) - 1;

        hbi->cur[i] = cur & (cur - 1);



        /* Set up next level for iteration.  */

        cur = hb->levels[i + 1][pos];

    }



    hbi->pos = pos;

    trace_hbitmap_iter_skip_words(hbi->hb, hbi, pos, cur);



    assert(cur);

    return cur;

}
