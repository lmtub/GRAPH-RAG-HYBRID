static av_cold void rv34_init_tables(void)

{

    int i, j, k;



    for(i = 0; i < NUM_INTRA_TABLES; i++){

        for(j = 0; j < 2; j++){

            rv34_gen_vlc(rv34_table_intra_cbppat   [i][j], CBPPAT_VLC_SIZE,   &intra_vlcs[i].cbppattern[j],     NULL, 19*i + 0 + j);

            rv34_gen_vlc(rv34_table_intra_secondpat[i][j], OTHERBLK_VLC_SIZE, &intra_vlcs[i].second_pattern[j], NULL, 19*i + 2 + j);

            rv34_gen_vlc(rv34_table_intra_thirdpat [i][j], OTHERBLK_VLC_SIZE, &intra_vlcs[i].third_pattern[j],  NULL, 19*i + 4 + j);

            for(k = 0; k < 4; k++){

                rv34_gen_vlc(rv34_table_intra_cbp[i][j+k*2],  CBP_VLC_SIZE,   &intra_vlcs[i].cbp[j][k],         rv34_cbp_code, 19*i + 6 + j*4 + k);

            }

        }

        for(j = 0; j < 4; j++){

            rv34_gen_vlc(rv34_table_intra_firstpat[i][j], FIRSTBLK_VLC_SIZE, &intra_vlcs[i].first_pattern[j], NULL, 19*i + 14 + j);

        }

        rv34_gen_vlc(rv34_intra_coeff[i], COEFF_VLC_SIZE, &intra_vlcs[i].coefficient, NULL, 19*i + 18);

    }



    for(i = 0; i < NUM_INTER_TABLES; i++){

        rv34_gen_vlc(rv34_inter_cbppat[i], CBPPAT_VLC_SIZE, &inter_vlcs[i].cbppattern[0], NULL, i*12 + 95);

        for(j = 0; j < 4; j++){

            rv34_gen_vlc(rv34_inter_cbp[i][j], CBP_VLC_SIZE, &inter_vlcs[i].cbp[0][j], rv34_cbp_code, i*12 + 96 + j);

        }

        for(j = 0; j < 2; j++){

            rv34_gen_vlc(rv34_table_inter_firstpat [i][j], FIRSTBLK_VLC_SIZE, &inter_vlcs[i].first_pattern[j],  NULL, i*12 + 100 + j);

            rv34_gen_vlc(rv34_table_inter_secondpat[i][j], OTHERBLK_VLC_SIZE, &inter_vlcs[i].second_pattern[j], NULL, i*12 + 102 + j);

            rv34_gen_vlc(rv34_table_inter_thirdpat [i][j], OTHERBLK_VLC_SIZE, &inter_vlcs[i].third_pattern[j],  NULL, i*12 + 104 + j);

        }

        rv34_gen_vlc(rv34_inter_coeff[i], COEFF_VLC_SIZE, &inter_vlcs[i].coefficient, NULL, i*12 + 106);

    }

}
