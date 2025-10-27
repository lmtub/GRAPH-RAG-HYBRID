static int bitplane_decoding(BitPlane *bp, VC9Context *v)

{

    GetBitContext *gb = &v->s.gb;



    int imode, x, y, code, use_vertical_tile, tile_w, tile_h;

    uint8_t invert, *planep = bp->data;



    invert = get_bits(gb, 1);

    imode = get_vlc2(gb, vc9_imode_vlc.table, VC9_IMODE_VLC_BITS, 2);



    bp->is_raw = 0;

    switch (imode)

    {

    case IMODE_RAW:

        //Data is actually read in the MB layer (same for all tests == "raw")

        bp->is_raw = 1; //invert ignored

        return invert;

    case IMODE_DIFF2:

    case IMODE_NORM2:

        if ((bp->height*bp->width) & 1) *(++planep) = get_bits(gb, 1);

        for(x=0; x<(bp->height*bp->width)>>1; x++){

            code = get_vlc2(gb, vc9_norm2_vlc.table, VC9_NORM2_VLC_BITS, 2);

            *(++planep) = code&1; //lsb => left

            *(++planep) = (code>>1)&1; //msb => right

            //FIXME width->stride

        }

        break;

    case IMODE_DIFF6:

    case IMODE_NORM6:

        use_vertical_tile=  bp->height%3==0 &&  bp->width%3!=0;

        tile_w= use_vertical_tile ? 2 : 3;

        tile_h= use_vertical_tile ? 3 : 2;



        for(y=  bp->height%tile_h; y< bp->height; y+=tile_h){

            for(x=  bp->width%tile_w; x< bp->width; x+=tile_w){

#if VLC_NORM6_METH0D == 1

                code = get_vlc2(gb, vc9_norm6_vlc.table, VC9_NORM6_VLC_BITS, 2);

                if(code<0){

                    av_log(v->s.avctx, AV_LOG_DEBUG, "invalid NORM-6 VLC\n");

                    return -1;

                }

#endif

#if VLC_NORM6_METH0D == 2 //Failure

                code = get_vlc2(gb, vc9_norm6_first_vlc.table, VC9_NORM6_FIRST_BITS, 2);

                if (code == 22)

                {

                    code = vc9_norm6_flc_val[get_bits(gb, 5)];

                }

                else if (code == 23)

                {

#  if TRACE

                    code = get_vlc2(gb, vc9_norm6_second_vlc.table, VC9_NORM6_SECOND_BITS, 2);

                    assert(code>-1 && code<22);

                    code = vc9_norm6_second_val[code];

#  else

                    code = vc9_norm6_second_val[get_vlc2(gb, vc9_norm6_second_vlc.table, VC9_NORM6_SECOND_BITS, 2)];

#  endif

                }

#endif //VLC_NORM6_METH0D == 2

                //FIXME following is a pure guess and probably wrong

                //FIXME A bitplane (0 | !0), so could the shifts be avoided ?

                planep[x     + 0*bp->stride]= (code>>0)&1;

                planep[x + 1 + 0*bp->stride]= (code>>1)&1;

                //FIXME Does branch prediction help here?

                if(use_vertical_tile){

                    planep[x + 0 + 1*bp->stride]= (code>>2)&1;

                    planep[x + 1 + 1*bp->stride]= (code>>3)&1;

                    planep[x + 0 + 2*bp->stride]= (code>>4)&1;

                    planep[x + 1 + 2*bp->stride]= (code>>5)&1;

                }else{

                    planep[x + 2 + 0*bp->stride]= (code>>2)&1;

                    planep[x + 0 + 1*bp->stride]= (code>>3)&1;

                    planep[x + 1 + 1*bp->stride]= (code>>4)&1;

                    planep[x + 2 + 1*bp->stride]= (code>>5)&1;

                }

            }

        }



        x=  bp->width % tile_w;

        decode_colskip(bp->data  ,             x, bp->height         , bp->stride, v);

        decode_rowskip(bp->data+x, bp->width - x, bp->height % tile_h, bp->stride, v);



        break;

    case IMODE_ROWSKIP:

        decode_rowskip(bp->data, bp->width, bp->height, bp->stride, v);

        break;

    case IMODE_COLSKIP: //Teh ugly

        decode_colskip(bp->data, bp->width, bp->height, bp->stride, v);

        break;

    default: break;

    }



    /* Applying diff operator */

    if (imode == IMODE_DIFF2 || imode == IMODE_DIFF6)

    {

        planep = bp->data;

        planep[0] ^= invert;

        for (x=1; x<bp->width; x++)

            planep[x] ^= planep[x-1];

        for (y=1; y<bp->height; y++)

        {

            planep += bp->stride;

            planep[0] ^= planep[-bp->stride];

            for (x=1; x<bp->width; x++)

            {

                if (planep[x-1] != planep[x-bp->stride]) planep[x] ^= invert;

                else                                     planep[x] ^= planep[x-1];

            }

        }

    }

    else if (invert)

    {

        planep = bp->data;

        for (x=0; x<bp->width*bp->height; x++) planep[x] = !planep[x]; //FIXME stride

    }

    return (imode<<1) + invert;

}
