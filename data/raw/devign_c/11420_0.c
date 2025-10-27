static void FUNCC(pred8x8_horizontal_add)(uint8_t *pix, const int *block_offset,

                                          const int16_t *block,

                                          ptrdiff_t stride)

{

    int i;

    for(i=0; i<4; i++)

        FUNCC(pred4x4_horizontal_add)(pix + block_offset[i], block + i*16*sizeof(pixel), stride);

}
