static inline int pic_is_unused(H264Context *h, Picture *pic)

{

    if (pic->f.data[0] == NULL)

        return 1;

    if (pic->needs_realloc && !(pic->reference & DELAYED_PIC_REF))

        return 1;

    return 0;

}
