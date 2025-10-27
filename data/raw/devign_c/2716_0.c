static inline void cow_set_bits(uint8_t *bitmap, int start, int64_t nb_sectors)

{

    int64_t bitnum = start, last = start + nb_sectors;

    while (bitnum < last) {

        if ((bitnum & 7) == 0 && bitnum + 8 <= last) {

            bitmap[bitnum / 8] = 0xFF;

            bitnum += 8;

            continue;

        }

        bitmap[bitnum/8] |= (1 << (bitnum % 8));

        bitnum++;

    }

}
