static void create_vorbis_context(venc_context_t * venc, AVCodecContext * avccontext) {

    codebook_t * cb;

    floor_t * fc;

    residue_t * rc;

    mapping_t * mc;

    int i, book;



    venc->channels = avccontext->channels;

    venc->sample_rate = avccontext->sample_rate;

    venc->blocksize[0] = venc->blocksize[1] = 8;



    venc->ncodebooks = 10;

    venc->codebooks = av_malloc(sizeof(codebook_t) * venc->ncodebooks);



    // codebook 1 - floor1 book, values 0..255

    cb = &venc->codebooks[0];

    cb->nentries = 256;

    cb->entries = av_malloc(sizeof(cb_entry_t) * cb->nentries);

    for (i = 0; i < cb->nentries; i++) cb->entries[i].len = 8;

    cb->ndimentions = 0;

    cb->min = 0.;

    cb->delta = 0.;

    cb->seq_p = 0;

    cb->lookup = 0;

    cb->quantlist = NULL;

    ready_codebook(cb);



    // codebook 2 - residue classbook, values 0..1, dimentions 200

    cb = &venc->codebooks[1];

    cb->nentries = 2;

    cb->entries = av_malloc(sizeof(cb_entry_t) * cb->nentries);

    for (i = 0; i < cb->nentries; i++) cb->entries[i].len = 1;

    cb->ndimentions = 200;

    cb->min = 0.;

    cb->delta = 0.;

    cb->seq_p = 0;

    cb->lookup = 0;

    cb->quantlist = NULL;

    ready_codebook(cb);



    // codebook 3..10 - vector, for the residue, values -32767..32767, dimentions 1

    for (book = 0; book < 8; book++) {

        cb = &venc->codebooks[2 + book];

        cb->nentries = 5;

        cb->entries = av_malloc(sizeof(cb_entry_t) * cb->nentries);

        for (i = 0; i < cb->nentries; i++) cb->entries[i].len = i == 2 ? 1 : 3;

        cb->ndimentions = 1;

        cb->delta = 1 << ((7 - book) * 2);

        cb->min = -cb->delta*2;

        cb->seq_p = 0;

        cb->lookup = 2;

        cb->quantlist = av_malloc(sizeof(int) * cb_lookup_vals(cb->lookup, cb->ndimentions, cb->nentries));

        for (i = 0; i < cb->nentries; i++) cb->quantlist[i] = i;

        ready_codebook(cb);

    }



    venc->nfloors = 1;

    venc->floors = av_malloc(sizeof(floor_t) * venc->nfloors);



    // just 1 floor

    fc = &venc->floors[0];

    fc->partitions = 1;

    fc->partition_to_class = av_malloc(sizeof(int) * fc->partitions);

    for (i = 0; i < fc->partitions; i++) fc->partition_to_class = 0;

    fc->nclasses = 1;

    fc->classes = av_malloc(sizeof(floor_class_t) * fc->nclasses);

    for (i = 0; i < fc->nclasses; i++) {

        floor_class_t * c = &fc->classes[i];

        int j, books;

        c->dim = 1;

        c->subclass = 0;

        c->masterbook = 0;

        books = (1 << c->subclass);

        c->books = av_malloc(sizeof(int) * books);

        for (j = 0; j < books; j++) c->books[j] = 0;

    }

    fc->multiplier = 1;

    fc->rangebits = venc->blocksize[0];



    fc->values = 2;

    for (i = 0; i < fc->partitions; i++)

        fc->values += fc->classes[fc->partition_to_class[i]].dim;



    fc->list = av_malloc(sizeof(*fc->list) * fc->values);

    fc->list[0].x = 0;

    fc->list[1].x = 1 << fc->rangebits;

    for (i = 2; i < fc->values; i++) fc->list[i].x = i * 5;



    venc->nresidues = 1;

    venc->residues = av_malloc(sizeof(residue_t) * venc->nresidues);



    // single residue

    rc = &venc->residues[0];

    rc->type = 0;

    rc->begin = 0;

    rc->end = 1 << venc->blocksize[0];

    rc->partition_size = 64;

    rc->classifications = 1;

    rc->classbook = 1;

    rc->books = av_malloc(sizeof(int[8]) * rc->classifications);

    for (i = 0; i < 8; i++) rc->books[0][i] = 2 + i;



    venc->nmappings = 1;

    venc->mappings = av_malloc(sizeof(mapping_t) * venc->nmappings);



    // single mapping

    mc = &venc->mappings[0];

    mc->submaps = 1;

    mc->mux = av_malloc(sizeof(int) * venc->channels);

    for (i = 0; i < venc->channels; i++) mc->mux[i] = 0;

    mc->floor = av_malloc(sizeof(int) * mc->submaps);

    mc->residue = av_malloc(sizeof(int) * mc->submaps);

    for (i = 0; i < mc->submaps; i++) {

        mc->floor[i] = 0;

        mc->residue[i] = 0;

    }



    venc->nmodes = 1;

    venc->modes = av_malloc(sizeof(vorbis_mode_t) * venc->nmodes);



    // single mode

    venc->modes[0].blockflag = 0;

    venc->modes[0].mapping = 0;

}
