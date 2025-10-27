static uint64_t hb_count_between(HBitmap *hb, uint64_t start, uint64_t last)

{

    HBitmapIter hbi;

    uint64_t count = 0;

    uint64_t end = last + 1;

    unsigned long cur;

    size_t pos;



    hbitmap_iter_init(&hbi, hb, start << hb->granularity);

    for (;;) {

        pos = hbitmap_iter_next_word(&hbi, &cur);

        if (pos >= (end >> BITS_PER_LEVEL)) {

            break;

        }

        count += popcountl(cur);

    }



    if (pos == (end >> BITS_PER_LEVEL)) {

        /* Drop bits representing the END-th and subsequent items.  */

        int bit = end & (BITS_PER_LONG - 1);

        cur &= (1UL << bit) - 1;

        count += popcountl(cur);

    }



    return count;

}
