static void floor_encode(venc_context_t * venc, floor_t * fc, PutBitContext * pb, int * posts, float * floor, int samples) {

    int range = 255 / fc->multiplier + 1;

    int coded[fc->values]; // first 2 values are unused

    int i, counter;

    int lx, ly;



    put_bits(pb, 1, 1); // non zero

    put_bits(pb, ilog(range - 1), posts[0]);

    put_bits(pb, ilog(range - 1), posts[1]);



    for (i = 2; i < fc->values; i++) {

        int predicted = render_point(fc->list[fc->list[i].low].x,

                                     posts[fc->list[i].low],

                                     fc->list[fc->list[i].high].x,

                                     posts[fc->list[i].high],

                                     fc->list[i].x);

        int highroom = range - predicted;

        int lowroom = predicted;

        int room = FFMIN(highroom, lowroom);

        if (predicted == posts[i]) {

            coded[i] = 0; // must be used later as flag!

            continue;

        } else {

            if (!coded[fc->list[i].low]) coded[fc->list[i].low] = -1;

            if (!coded[fc->list[i].high]) coded[fc->list[i].high] = -1;

        }

        if (posts[i] > predicted) {

            if (posts[i] - predicted > room) coded[i] = posts[i] - predicted + lowroom;

            else coded[i] = (posts[i] - predicted) << 1;

        } else {

            if (predicted - posts[i] > room) coded[i] = predicted - posts[i] + highroom - 1;

            else coded[i] = ((predicted - posts[i]) << 1) - 1;

        }

    }



    counter = 2;

    for (i = 0; i < fc->partitions; i++) {

        floor_class_t * c = &fc->classes[fc->partition_to_class[i]];

        int k, cval = 0, csub = 1<<c->subclass;

        if (c->subclass) {

            codebook_t * book = &venc->codebooks[c->masterbook];

            int cshift = 0;

            for (k = 0; k < c->dim; k++) {

                int l;

                for (l = 0; l < csub; l++) {

                    int maxval = 1;

                    if (c->books[l] != -1) maxval = venc->codebooks[c->books[l]].nentries;

                    // coded could be -1, but this still works, cause thats 0

                    if (coded[counter + k] < maxval) break;

                }

                assert(l != csub);

                cval |= l << cshift;

                cshift += c->subclass;

            }

            assert(cval < book->nentries);

            put_bits(pb, book->entries[cval].len, book->entries[cval].codeword);

        }

        for (k = 0; k < c->dim; k++) {

            int book = c->books[cval & (csub-1)];

            int entry = coded[counter++];

            cval >>= c->subclass;

            if (book == -1) continue;

            if (entry == -1) entry = 0;

            assert(entry < venc->codebooks[book].nentries);

            assert(entry >= 0);

            put_bits(pb, venc->codebooks[book].entries[entry].len, venc->codebooks[book].entries[entry].codeword);

        }

    }



    lx = 0;

    ly = posts[0] * fc->multiplier; // sorted 0 is still 0

    coded[0] = coded[1] = 1;

    for (i = 1; i < fc->values; i++) {

        int pos = fc->list[i].sort;

        if (coded[pos]) {

            render_line(lx, ly, fc->list[pos].x, posts[pos] * fc->multiplier, floor, samples);

            lx = fc->list[pos].x;

            ly = posts[pos] * fc->multiplier;

        }

        if (lx >= samples) break;

    }

    if (lx < samples) render_line(lx, ly, samples, ly, floor, samples);

}
