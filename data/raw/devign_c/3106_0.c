void avg_pixels8_altivec(uint8_t * block, const uint8_t * pixels, int line_size, int h)

{

POWERPC_TBL_DECLARE(altivec_avg_pixels8_num, 1);

#ifdef ALTIVEC_USE_REFERENCE_C_CODE

    int i;

POWERPC_TBL_START_COUNT(altivec_avg_pixels8_num, 1);

    for (i = 0; i < h; i++) {

        *((uint32_t *) (block)) =

            (((*((uint32_t *) (block))) |

              ((((const struct unaligned_32 *) (pixels))->l))) -

             ((((*((uint32_t *) (block))) ^

                ((((const struct unaligned_32 *) (pixels))->

                  l))) & 0xFEFEFEFEUL) >> 1));

        *((uint32_t *) (block + 4)) =

            (((*((uint32_t *) (block + 4))) |

              ((((const struct unaligned_32 *) (pixels + 4))->l))) -

             ((((*((uint32_t *) (block + 4))) ^

                ((((const struct unaligned_32 *) (pixels +

                                                  4))->

                  l))) & 0xFEFEFEFEUL) >> 1));

        pixels += line_size;

        block += line_size;

    }

POWERPC_TBL_STOP_COUNT(altivec_avg_pixels8_num, 1);



#else /* ALTIVEC_USE_REFERENCE_C_CODE */

    register vector unsigned char pixelsv1, pixelsv2, pixelsv, blockv;

    int i;



POWERPC_TBL_START_COUNT(altivec_avg_pixels8_num, 1);

 

   for (i = 0; i < h; i++) {

     /*

       block is 8 bytes-aligned, so we're either in the

       left block (16 bytes-aligned) or in the right block (not)

     */

     int rightside = ((unsigned long)block & 0x0000000F);

     

     blockv = vec_ld(0, block);

     pixelsv1 = vec_ld(0, (unsigned char*)pixels);

     pixelsv2 = vec_ld(16, (unsigned char*)pixels);

     pixelsv = vec_perm(pixelsv1, pixelsv2, vec_lvsl(0, pixels));

     

     if (rightside)

     {

       pixelsv = vec_perm(blockv, pixelsv, vcprm(0,1,s0,s1));

     }

     else

     {

       pixelsv = vec_perm(blockv, pixelsv, vcprm(s0,s1,2,3));

     }

     

     blockv = vec_avg(blockv, pixelsv);



     vec_st(blockv, 0, block);

     

     pixels += line_size;

     block += line_size;

   }

   

POWERPC_TBL_STOP_COUNT(altivec_avg_pixels8_num, 1);

 

#endif /* ALTIVEC_USE_REFERENCE_C_CODE */

}
