void put_no_rnd_pixels8_xy2_altivec(uint8_t *block, const uint8_t *pixels, int line_size, int h)

{

POWERPC_TBL_DECLARE(altivec_put_no_rnd_pixels8_xy2_num, 1);

#ifdef ALTIVEC_USE_REFERENCE_C_CODE

    int j;

POWERPC_TBL_START_COUNT(altivec_put_no_rnd_pixels8_xy2_num, 1);

    for (j = 0; j < 2; j++) {

      int i;

      const uint32_t a = (((const struct unaligned_32 *) (pixels))->l);

      const uint32_t b =

        (((const struct unaligned_32 *) (pixels + 1))->l);

      uint32_t l0 =

        (a & 0x03030303UL) + (b & 0x03030303UL) + 0x01010101UL;

      uint32_t h0 =

        ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);

      uint32_t l1, h1;

      pixels += line_size;

      for (i = 0; i < h; i += 2) {

        uint32_t a = (((const struct unaligned_32 *) (pixels))->l);

        uint32_t b = (((const struct unaligned_32 *) (pixels + 1))->l);

        l1 = (a & 0x03030303UL) + (b & 0x03030303UL);

        h1 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);

        *((uint32_t *) block) =

          h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL);

        pixels += line_size;

        block += line_size;

        a = (((const struct unaligned_32 *) (pixels))->l);

        b = (((const struct unaligned_32 *) (pixels + 1))->l);

        l0 = (a & 0x03030303UL) + (b & 0x03030303UL) + 0x01010101UL;

        h0 = ((a & 0xFCFCFCFCUL) >> 2) + ((b & 0xFCFCFCFCUL) >> 2);

        *((uint32_t *) block) =

          h0 + h1 + (((l0 + l1) >> 2) & 0x0F0F0F0FUL);

        pixels += line_size;

        block += line_size;

      } pixels += 4 - line_size * (h + 1);

      block += 4 - line_size * h;

    }

    

POWERPC_TBL_STOP_COUNT(altivec_put_no_rnd_pixels8_xy2_num, 1);



#else /* ALTIVEC_USE_REFERENCE_C_CODE */

   register int i;

   register vector unsigned char

     pixelsv1, pixelsv2,

     pixelsavg;

   register vector unsigned char

     blockv, temp1, temp2;

   register vector unsigned short

     pixelssum1, pixelssum2, temp3;

   register const vector unsigned char vczero = (const vector unsigned char)vec_splat_u8(0);

   register const vector unsigned short vcone = (const vector unsigned short)vec_splat_u16(1);

   register const vector unsigned short vctwo = (const vector unsigned short)vec_splat_u16(2);

   

   temp1 = vec_ld(0, pixels);

   temp2 = vec_ld(16, pixels);

   pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(0, pixels));

   if ((((unsigned long)pixels) & 0x0000000F) ==  0x0000000F)

   {

     pixelsv2 = temp2;

   }

   else

   {

     pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(1, pixels));

   }

   pixelsv1 = vec_mergeh(vczero, pixelsv1);

   pixelsv2 = vec_mergeh(vczero, pixelsv2);

   pixelssum1 = vec_add((vector unsigned short)pixelsv1,

                        (vector unsigned short)pixelsv2);

   pixelssum1 = vec_add(pixelssum1, vcone);

   

POWERPC_TBL_START_COUNT(altivec_put_no_rnd_pixels8_xy2_num, 1); 

   for (i = 0; i < h ; i++) {

     int rightside = ((unsigned long)block & 0x0000000F);

     blockv = vec_ld(0, block);



     temp1 = vec_ld(line_size, pixels);

     temp2 = vec_ld(line_size + 16, pixels);

     pixelsv1 = vec_perm(temp1, temp2, vec_lvsl(line_size, pixels));

     if (((((unsigned long)pixels) + line_size) & 0x0000000F) ==  0x0000000F)

     {

       pixelsv2 = temp2;

     }

     else

     {

       pixelsv2 = vec_perm(temp1, temp2, vec_lvsl(line_size + 1, pixels));

     }



     pixelsv1 = vec_mergeh(vczero, pixelsv1);

     pixelsv2 = vec_mergeh(vczero, pixelsv2);

     pixelssum2 = vec_add((vector unsigned short)pixelsv1,

                          (vector unsigned short)pixelsv2);

     temp3 = vec_add(pixelssum1, pixelssum2);

     temp3 = vec_sra(temp3, vctwo);

     pixelssum1 = vec_add(pixelssum2, vcone);

     pixelsavg = vec_packsu(temp3, (vector unsigned short) vczero);

     

     if (rightside)

     {

       blockv = vec_perm(blockv, pixelsavg, vcprm(0, 1, s0, s1));

     }

     else

     {

       blockv = vec_perm(blockv, pixelsavg, vcprm(s0, s1, 2, 3));

     }

     

     vec_st(blockv, 0, block);

     

     block += line_size;

     pixels += line_size;

   }

   

POWERPC_TBL_STOP_COUNT(altivec_put_no_rnd_pixels8_xy2_num, 1);

#endif /* ALTIVEC_USE_REFERENCE_C_CODE */

}
