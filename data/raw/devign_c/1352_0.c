static int raw_pread(BlockDriverState *bs, int64_t offset,

                     uint8_t *buf, int count)

{

    BDRVRawState *s = bs->opaque;

    int size, ret, shift, sum;



    sum = 0;



    if (s->aligned_buf != NULL)  {



        if (offset & 0x1ff) {

            /* align offset on a 512 bytes boundary */



            shift = offset & 0x1ff;

            size = (shift + count + 0x1ff) & ~0x1ff;

            if (size > ALIGNED_BUFFER_SIZE)

                size = ALIGNED_BUFFER_SIZE;

            ret = raw_pread_aligned(bs, offset - shift, s->aligned_buf, size);

            if (ret < 0)

                return ret;



            size = 512 - shift;

            if (size > count)

                size = count;

            memcpy(buf, s->aligned_buf + shift, size);



            buf += size;

            offset += size;

            count -= size;

            sum += size;



            if (count == 0)

                return sum;

        }

        if (count & 0x1ff || (uintptr_t) buf & 0x1ff) {



            /* read on aligned buffer */



            while (count) {



                size = (count + 0x1ff) & ~0x1ff;

                if (size > ALIGNED_BUFFER_SIZE)

                    size = ALIGNED_BUFFER_SIZE;



                ret = raw_pread_aligned(bs, offset, s->aligned_buf, size);

                if (ret < 0) {

                    return ret;

                } else if (ret == 0) {

                    fprintf(stderr, "raw_pread: read beyond end of file\n");

                    abort();

                }



                size = ret;

                if (size > count)

                    size = count;



                memcpy(buf, s->aligned_buf, size);



                buf += size;

                offset += size;

                count -= size;

                sum += size;

            }



            return sum;

        }

    }



    return raw_pread_aligned(bs, offset, buf, count) + sum;

}
