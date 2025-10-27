static int mov_read_tkhd(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    int i;

    int width;

    int height;

    int display_matrix[3][3];

    AVStream *st;

    MOVStreamContext *sc;

    int version;

    int flags;



    if (c->fc->nb_streams < 1)

        return 0;

    st = c->fc->streams[c->fc->nb_streams-1];

    sc = st->priv_data;



    version = avio_r8(pb);

    flags = avio_rb24(pb);

    st->disposition |= (flags & MOV_TKHD_FLAG_ENABLED) ? AV_DISPOSITION_DEFAULT : 0;



    if (version == 1) {

        avio_rb64(pb);

        avio_rb64(pb);

    } else {

        avio_rb32(pb); /* creation time */

        avio_rb32(pb); /* modification time */

    }

    st->id = (int)avio_rb32(pb); /* track id (NOT 0 !)*/

    avio_rb32(pb); /* reserved */



    /* highlevel (considering edits) duration in movie timebase */

    (version == 1) ? avio_rb64(pb) : avio_rb32(pb);

    avio_rb32(pb); /* reserved */

    avio_rb32(pb); /* reserved */



    avio_rb16(pb); /* layer */

    avio_rb16(pb); /* alternate group */

    avio_rb16(pb); /* volume */

    avio_rb16(pb); /* reserved */



    //read in the display matrix (outlined in ISO 14496-12, Section 6.2.2)

    // they're kept in fixed point format through all calculations

    // save u,v,z to store the whole matrix in the AV_PKT_DATA_DISPLAYMATRIX

    // side data, but the scale factor is not needed to calculate aspect ratio

    for (i = 0; i < 3; i++) {

        display_matrix[i][0] = avio_rb32(pb);   // 16.16 fixed point

        display_matrix[i][1] = avio_rb32(pb);   // 16.16 fixed point

        display_matrix[i][2] = avio_rb32(pb);   //  2.30 fixed point

    }



    width = avio_rb32(pb);       // 16.16 fixed point track width

    height = avio_rb32(pb);      // 16.16 fixed point track height

    sc->width = width >> 16;

    sc->height = height >> 16;



    // save the matrix and add rotate metadata when it is not the default

    // identity

    if (display_matrix[0][0] != (1 << 16) ||

        display_matrix[1][1] != (1 << 16) ||

        display_matrix[2][2] != (1 << 30) ||

        display_matrix[0][1] || display_matrix[0][2] ||

        display_matrix[1][0] || display_matrix[1][2] ||

        display_matrix[2][0] || display_matrix[2][1]) {

        int i, j;

        double rotate;



        av_freep(&sc->display_matrix);

        sc->display_matrix = av_malloc(sizeof(int32_t) * 9);

        if (!sc->display_matrix)

            return AVERROR(ENOMEM);



        for (i = 0; i < 3; i++)

            for (j = 0; j < 3; j++)

                sc->display_matrix[i * 3 + j] = display_matrix[j][i];



        rotate = av_display_rotation_get(sc->display_matrix);

        if (!isnan(rotate)) {

            char rotate_buf[64];

            rotate = -rotate;

            if (rotate < 0) // for backward compatibility

                rotate += 360;

            snprintf(rotate_buf, sizeof(rotate_buf), "%g", rotate);

            av_dict_set(&st->metadata, "rotate", rotate_buf, 0);

        }

    }



    // transform the display width/height according to the matrix

    // to keep the same scale, use [width height 1<<16]

    if (width && height && sc->display_matrix) {

        double disp_transform[2];



#define SQR(a) ((a)*(double)(a))

        for (i = 0; i < 2; i++)

            disp_transform[i] = sqrt(SQR(display_matrix[i][0]) + SQR(display_matrix[i][1]));



        if (disp_transform[0] > 0 && disp_transform[1] > 0 &&

            fabs((disp_transform[0] / disp_transform[1]) - 1.0) > 0.01)

            st->sample_aspect_ratio = av_d2q(

                disp_transform[0] / disp_transform[1],

                INT_MAX);

    }

    return 0;

}
