mp_image_t* vf_get_image(vf_instance_t* vf, unsigned int outfmt, int mp_imgtype, int mp_imgflag, int w, int h){

    MPContext *m= (MPContext*)(((uint8_t*)vf) - offsetof(MPContext, next_vf));

  mp_image_t* mpi=NULL;

  int w2;

  int number = mp_imgtype >> 16;



  av_assert0(vf->next == NULL); // all existing filters call this just on next



  //vf_dint needs these as it calls vf_get_image() before configuring the output

  if(vf->w==0 && w>0) vf->w=w;

  if(vf->h==0 && h>0) vf->h=h;



  av_assert0(w == -1 || w >= vf->w);

  av_assert0(h == -1 || h >= vf->h);

  av_assert0(vf->w > 0);

  av_assert0(vf->h > 0);



  av_log(m->avfctx, AV_LOG_DEBUG, "get_image: %d:%d, vf: %d:%d\n", w,h,vf->w,vf->h);



  if (w == -1) w = vf->w;

  if (h == -1) h = vf->h;



  w2=(mp_imgflag&MP_IMGFLAG_ACCEPT_ALIGNED_STRIDE)?((w+15)&(~15)):w;



  // Note: we should call libvo first to check if it supports direct rendering

  // and if not, then fallback to software buffers:

  switch(mp_imgtype & 0xff){

  case MP_IMGTYPE_EXPORT:

    if(!vf->imgctx.export_images[0]) vf->imgctx.export_images[0]=new_mp_image(w2,h);

    mpi=vf->imgctx.export_images[0];

    break;

  case MP_IMGTYPE_STATIC:

    if(!vf->imgctx.static_images[0]) vf->imgctx.static_images[0]=new_mp_image(w2,h);

    mpi=vf->imgctx.static_images[0];

    break;

  case MP_IMGTYPE_TEMP:

    if(!vf->imgctx.temp_images[0]) vf->imgctx.temp_images[0]=new_mp_image(w2,h);

    mpi=vf->imgctx.temp_images[0];

    break;

  case MP_IMGTYPE_IPB:

    if(!(mp_imgflag&MP_IMGFLAG_READABLE)){ // B frame:

      if(!vf->imgctx.temp_images[0]) vf->imgctx.temp_images[0]=new_mp_image(w2,h);

      mpi=vf->imgctx.temp_images[0];

      break;

    }

  case MP_IMGTYPE_IP:

    if(!vf->imgctx.static_images[vf->imgctx.static_idx]) vf->imgctx.static_images[vf->imgctx.static_idx]=new_mp_image(w2,h);

    mpi=vf->imgctx.static_images[vf->imgctx.static_idx];

    vf->imgctx.static_idx^=1;

    break;

  case MP_IMGTYPE_NUMBERED:

    if (number == -1) {

      int i;

      for (i = 0; i < NUM_NUMBERED_MPI; i++)

        if (!vf->imgctx.numbered_images[i] || !vf->imgctx.numbered_images[i]->usage_count)

          break;

      number = i;

    }

    if (number < 0 || number >= NUM_NUMBERED_MPI) return NULL;

    if (!vf->imgctx.numbered_images[number]) vf->imgctx.numbered_images[number] = new_mp_image(w2,h);

    mpi = vf->imgctx.numbered_images[number];

    mpi->number = number;

    break;

  }

  if(mpi){

    mpi->type=mp_imgtype;

    mpi->w=vf->w; mpi->h=vf->h;

    // keep buffer allocation status & color flags only:

//    mpi->flags&=~(MP_IMGFLAG_PRESERVE|MP_IMGFLAG_READABLE|MP_IMGFLAG_DIRECT);

    mpi->flags&=MP_IMGFLAG_ALLOCATED|MP_IMGFLAG_TYPE_DISPLAYED|MP_IMGFLAGMASK_COLORS;

    // accept restrictions, draw_slice and palette flags only:

    mpi->flags|=mp_imgflag&(MP_IMGFLAGMASK_RESTRICTIONS|MP_IMGFLAG_DRAW_CALLBACK|MP_IMGFLAG_RGB_PALETTE);

    if(!vf->draw_slice) mpi->flags&=~MP_IMGFLAG_DRAW_CALLBACK;

    if(mpi->width!=w2 || mpi->height!=h){

//      printf("vf.c: MPI parameters changed!  %dx%d -> %dx%d   \n", mpi->width,mpi->height,w2,h);

        if(mpi->flags&MP_IMGFLAG_ALLOCATED){

            if(mpi->width<w2 || mpi->height<h){

                // need to re-allocate buffer memory:

                av_free(mpi->planes[0]);

                mpi->flags&=~MP_IMGFLAG_ALLOCATED;

                mp_msg(MSGT_VFILTER,MSGL_V,"vf.c: have to REALLOCATE buffer memory :(\n");

            }

//      } else {

        } {

            mpi->width=w2; mpi->chroma_width=(w2 + (1<<mpi->chroma_x_shift) - 1)>>mpi->chroma_x_shift;

            mpi->height=h; mpi->chroma_height=(h + (1<<mpi->chroma_y_shift) - 1)>>mpi->chroma_y_shift;

        }

    }

    if(!mpi->bpp) mp_image_setfmt(mpi,outfmt);

    if(!(mpi->flags&MP_IMGFLAG_ALLOCATED) && mpi->type>MP_IMGTYPE_EXPORT){



        av_assert0(!vf->get_image);

        // check libvo first!

        if(vf->get_image) vf->get_image(vf,mpi);



        if(!(mpi->flags&MP_IMGFLAG_DIRECT)){

          // non-direct and not yet allocated image. allocate it!

          if (!mpi->bpp) { // no way we can allocate this

              mp_msg(MSGT_DECVIDEO, MSGL_FATAL,

                     "vf_get_image: Tried to allocate a format that can not be allocated!\n");

              return NULL;

          }



          // check if codec prefer aligned stride:

          if(mp_imgflag&MP_IMGFLAG_PREFER_ALIGNED_STRIDE){

              int align=(mpi->flags&MP_IMGFLAG_PLANAR &&

                         mpi->flags&MP_IMGFLAG_YUV) ?

                         (8<<mpi->chroma_x_shift)-1 : 15; // -- maybe FIXME

              w2=((w+align)&(~align));

              if(mpi->width!=w2){

#if 0

                  // we have to change width... check if we CAN co it:

                  int flags=vf->query_format(vf,outfmt); // should not fail

                  if(!(flags&3)) mp_msg(MSGT_DECVIDEO,MSGL_WARN,"??? vf_get_image{vf->query_format(outfmt)} failed!\n");

//                printf("query -> 0x%X    \n",flags);

                  if(flags&VFCAP_ACCEPT_STRIDE){

#endif

                      mpi->width=w2;

                      mpi->chroma_width=(w2 + (1<<mpi->chroma_x_shift) - 1)>>mpi->chroma_x_shift;

//                  }

              }

          }



          mp_image_alloc_planes(mpi);

//        printf("clearing img!\n");

          vf_mpi_clear(mpi,0,0,mpi->width,mpi->height);

        }

    }

    av_assert0(!vf->start_slice);

    if(mpi->flags&MP_IMGFLAG_DRAW_CALLBACK)

        if(vf->start_slice) vf->start_slice(vf,mpi);

    if(!(mpi->flags&MP_IMGFLAG_TYPE_DISPLAYED)){

            mp_msg(MSGT_DECVIDEO,MSGL_V,"*** [%s] %s%s mp_image_t, %dx%dx%dbpp %s %s, %d bytes\n",

                  "NULL"/*vf->info->name*/,

                  (mpi->type==MP_IMGTYPE_EXPORT)?"Exporting":

                  ((mpi->flags&MP_IMGFLAG_DIRECT)?"Direct Rendering":"Allocating"),

                  (mpi->flags&MP_IMGFLAG_DRAW_CALLBACK)?" (slices)":"",

                  mpi->width,mpi->height,mpi->bpp,

                  (mpi->flags&MP_IMGFLAG_YUV)?"YUV":((mpi->flags&MP_IMGFLAG_SWAPPED)?"BGR":"RGB"),

                  (mpi->flags&MP_IMGFLAG_PLANAR)?"planar":"packed",

                  mpi->bpp*mpi->width*mpi->height/8);

            mp_msg(MSGT_DECVIDEO,MSGL_DBG2,"(imgfmt: %x, planes: %p,%p,%p strides: %d,%d,%d, chroma: %dx%d, shift: h:%d,v:%d)\n",

                mpi->imgfmt, mpi->planes[0], mpi->planes[1], mpi->planes[2],

                mpi->stride[0], mpi->stride[1], mpi->stride[2],

                mpi->chroma_width, mpi->chroma_height, mpi->chroma_x_shift, mpi->chroma_y_shift);

            mpi->flags|=MP_IMGFLAG_TYPE_DISPLAYED;

    }



  mpi->qscale = NULL;

  }

  mpi->usage_count++;

//    printf("\rVF_MPI: %p %p %p %d %d %d    \n",

//      mpi->planes[0],mpi->planes[1],mpi->planes[2],

//      mpi->stride[0],mpi->stride[1],mpi->stride[2]);

  return mpi;

}
