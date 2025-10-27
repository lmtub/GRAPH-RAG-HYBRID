static void fill_colmap(H264Context *h, int map[2][16+32], int list, int field, int colfield, int mbafi){

    MpegEncContext * const s = &h->s;

    Picture * const ref1 = &h->ref_list[1][0];

    int j, old_ref, rfield;

    int start= mbafi ? 16                      : 0;

    int end  = mbafi ? 16+2*h->ref_count[0]    : h->ref_count[0];

    int interl= mbafi || s->picture_structure != PICT_FRAME;



    /* bogus; fills in for missing frames */

    memset(map[list], 0, sizeof(map[list]));



    for(rfield=0; rfield<2; rfield++){

        for(old_ref=0; old_ref<ref1->ref_count[colfield][list]; old_ref++){

            int poc = ref1->ref_poc[colfield][list][old_ref];



            if     (!interl)

                poc |= 3;

            else if( interl && (poc&3) == 3) //FIXME store all MBAFF references so this isnt needed

                poc= (poc&~3) + rfield + 1;



            for(j=start; j<end; j++){

                if (4 * h->ref_list[0][j].frame_num + (h->ref_list[0][j].f.reference & 3) == poc) {

                    int cur_ref= mbafi ? (j-16)^field : j;

                    map[list][2*old_ref + (rfield^field) + 16] = cur_ref;

                    if(rfield == field || !interl)

                        map[list][old_ref] = cur_ref;

                    break;

                }

            }

        }

    }

}
