static void create_vorbis_context(vorbis_enc_context *venc,

                                  AVCodecContext *avccontext)

{

    vorbis_enc_floor   *fc;

    vorbis_enc_residue *rc;

    vorbis_enc_mapping *mc;

    int i, book;



    venc->channels    = avccontext->channels;

    venc->sample_rate = avccontext->sample_rate;

    venc->log2_blocksize[0] = venc->log2_blocksize[1] = 11;



    venc->ncodebooks = FF_ARRAY_ELEMS(cvectors);

    venc->codebooks  = av_malloc(sizeof(vorbis_enc_codebook) * venc->ncodebooks);



    // codebook 0..14 - floor1 book, values 0..255

    // codebook 15 residue masterbook

    // codebook 16..29 residue

    for (book = 0; book < venc->ncodebooks; book++) {

        vorbis_enc_codebook *cb = &venc->codebooks[book];

        int vals;

        cb->ndimentions = cvectors[book].dim;

        cb->nentries    = cvectors[book].real_len;

        cb->min         = cvectors[book].min;

        cb->delta       = cvectors[book].delta;

        cb->lookup      = cvectors[book].lookup;

        cb->seq_p       = 0;



        cb->lens      = av_malloc(sizeof(uint8_t)  * cb->nentries);

        cb->codewords = av_malloc(sizeof(uint32_t) * cb->nentries);

        memcpy(cb->lens, cvectors[book].clens, cvectors[book].len);

        memset(cb->lens + cvectors[book].len, 0, cb->nentries - cvectors[book].len);



        if (cb->lookup) {

            vals = cb_lookup_vals(cb->lookup, cb->ndimentions, cb->nentries);

            cb->quantlist = av_malloc(sizeof(int) * vals);

            for (i = 0; i < vals; i++)

                cb->quantlist[i] = cvectors[book].quant[i];

        } else {

            cb->quantlist = NULL;

        }

        ready_codebook(cb);

    }



    venc->nfloors = 1;

    venc->floors  = av_malloc(sizeof(vorbis_enc_floor) * venc->nfloors);



    // just 1 floor

    fc = &venc->floors[0];

    fc->partitions         = NUM_FLOOR_PARTITIONS;

    fc->partition_to_class = av_malloc(sizeof(int) * fc->partitions);

    fc->nclasses           = 0;

    for (i = 0; i < fc->partitions; i++) {

        static const int a[] = {0, 1, 2, 2, 3, 3, 4, 4};

        fc->partition_to_class[i] = a[i];

        fc->nclasses = FFMAX(fc->nclasses, fc->partition_to_class[i]);

    }

    fc->nclasses++;

    fc->classes = av_malloc(sizeof(vorbis_enc_floor_class) * fc->nclasses);

    for (i = 0; i < fc->nclasses; i++) {

        vorbis_enc_floor_class * c = &fc->classes[i];

        int j, books;

        c->dim        = floor_classes[i].dim;

        c->subclass   = floor_classes[i].subclass;

        c->masterbook = floor_classes[i].masterbook;

        books         = (1 << c->subclass);

        c->books      = av_malloc(sizeof(int) * books);

        for (j = 0; j < books; j++)

            c->books[j] = floor_classes[i].nbooks[j];

    }

    fc->multiplier = 2;

    fc->rangebits  = venc->log2_blocksize[0] - 1;



    fc->values = 2;

    for (i = 0; i < fc->partitions; i++)

        fc->values += fc->classes[fc->partition_to_class[i]].dim;



    fc->list = av_malloc(sizeof(vorbis_floor1_entry) * fc->values);

    fc->list[0].x = 0;

    fc->list[1].x = 1 << fc->rangebits;

    for (i = 2; i < fc->values; i++) {

        static const int a[] = {

             93, 23,372,  6, 46,186,750, 14, 33, 65,

            130,260,556,  3, 10, 18, 28, 39, 55, 79,

            111,158,220,312,464,650,850

        };

        fc->list[i].x = a[i - 2];

    }

    ff_vorbis_ready_floor1_list(fc->list, fc->values);



    venc->nresidues = 1;

    venc->residues  = av_malloc(sizeof(vorbis_enc_residue) * venc->nresidues);



    // single residue

    rc = &venc->residues[0];

    rc->type            = 2;

    rc->begin           = 0;

    rc->end             = 1600;

    rc->partition_size  = 32;

    rc->classifications = 10;

    rc->classbook       = 15;

    rc->books           = av_malloc(sizeof(*rc->books) * rc->classifications);

    {

        static const int8_t a[10][8] = {

            { -1, -1, -1, -1, -1, -1, -1, -1, },

            { -1, -1, 16, -1, -1, -1, -1, -1, },

            { -1, -1, 17, -1, -1, -1, -1, -1, },

            { -1, -1, 18, -1, -1, -1, -1, -1, },

            { -1, -1, 19, -1, -1, -1, -1, -1, },

            { -1, -1, 20, -1, -1, -1, -1, -1, },

            { -1, -1, 21, -1, -1, -1, -1, -1, },

            { 22, 23, -1, -1, -1, -1, -1, -1, },

            { 24, 25, -1, -1, -1, -1, -1, -1, },

            { 26, 27, 28, -1, -1, -1, -1, -1, },

        };

        memcpy(rc->books, a, sizeof a);

    }

    ready_residue(rc, venc);



    venc->nmappings = 1;

    venc->mappings  = av_malloc(sizeof(vorbis_enc_mapping) * venc->nmappings);



    // single mapping

    mc = &venc->mappings[0];

    mc->submaps = 1;

    mc->mux     = av_malloc(sizeof(int) * venc->channels);

    for (i = 0; i < venc->channels; i++)

        mc->mux[i] = 0;

    mc->floor   = av_malloc(sizeof(int) * mc->submaps);

    mc->residue = av_malloc(sizeof(int) * mc->submaps);

    for (i = 0; i < mc->submaps; i++) {

        mc->floor[i]   = 0;

        mc->residue[i] = 0;

    }

    mc->coupling_steps = venc->channels == 2 ? 1 : 0;

    mc->magnitude      = av_malloc(sizeof(int) * mc->coupling_steps);

    mc->angle          = av_malloc(sizeof(int) * mc->coupling_steps);

    if (mc->coupling_steps) {

        mc->magnitude[0] = 0;

        mc->angle[0]     = 1;

    }



    venc->nmodes = 1;

    venc->modes  = av_malloc(sizeof(vorbis_enc_mode) * venc->nmodes);



    // single mode

    venc->modes[0].blockflag = 0;

    venc->modes[0].mapping   = 0;



    venc->have_saved = 0;

    venc->saved      = av_malloc(sizeof(float) * venc->channels * (1 << venc->log2_blocksize[1]) / 2);

    venc->samples    = av_malloc(sizeof(float) * venc->channels * (1 << venc->log2_blocksize[1]));

    venc->floor      = av_malloc(sizeof(float) * venc->channels * (1 << venc->log2_blocksize[1]) / 2);

    venc->coeffs     = av_malloc(sizeof(float) * venc->channels * (1 << venc->log2_blocksize[1]) / 2);



    venc->win[0] = ff_vorbis_vwin[venc->log2_blocksize[0] - 6];

    venc->win[1] = ff_vorbis_vwin[venc->log2_blocksize[1] - 6];



    ff_mdct_init(&venc->mdct[0], venc->log2_blocksize[0], 0, 1.0);

    ff_mdct_init(&venc->mdct[1], venc->log2_blocksize[1], 0, 1.0);

}
