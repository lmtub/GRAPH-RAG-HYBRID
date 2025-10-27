static unsigned int mszh_decomp(const unsigned char * srcptr, int srclen, unsigned char * destptr, unsigned int destsize)

{

    unsigned char *destptr_bak = destptr;

    unsigned char *destptr_end = destptr + destsize;

    const unsigned char *srcptr_end = srcptr + srclen;

    unsigned mask = *srcptr++;

    unsigned maskbit = 0x80;



    while (srcptr < srcptr_end && destptr < destptr_end) {

        if (!(mask & maskbit)) {

            memcpy(destptr, srcptr, 4);

            destptr += 4;

            srcptr += 4;

        } else {

            unsigned ofs = bytestream_get_le16(&srcptr);

            unsigned cnt = (ofs >> 11) + 1;

            ofs &= 0x7ff;

            ofs = FFMIN(ofs, destptr - destptr_bak);

            cnt *= 4;

            cnt = FFMIN(cnt, destptr_end - destptr);

            av_memcpy_backptr(destptr, ofs, cnt);

            destptr += cnt;

        }

        maskbit >>= 1;

        if (!maskbit) {

            mask = *srcptr++;

            while (!mask) {

                if (destptr_end - destptr < 32 || srcptr_end - srcptr < 32) break;

                memcpy(destptr, srcptr, 32);

                destptr += 32;

                srcptr += 32;

                mask = *srcptr++;

            }

            maskbit = 0x80;

        }

    }



    return destptr - destptr_bak;

}
