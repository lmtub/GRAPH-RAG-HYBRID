static void reset_contexts(SnowContext *s){

    int plane_index, level, orientation;



    for(plane_index=0; plane_index<2; plane_index++){

        for(level=0; level<s->spatial_decomposition_count; level++){

            for(orientation=level ? 1:0; orientation<4; orientation++){

                memset(s->plane[plane_index].band[level][orientation].state, 0, sizeof(s->plane[plane_index].band[level][orientation].state));

            }

        }

    }

    memset(s->mb_band.state, 0, sizeof(s->mb_band.state));

    memset(s->mv_band[0].state, 0, sizeof(s->mv_band[0].state));

    memset(s->mv_band[1].state, 0, sizeof(s->mv_band[1].state));

    memset(s->header_state, 0, sizeof(s->header_state));

}
