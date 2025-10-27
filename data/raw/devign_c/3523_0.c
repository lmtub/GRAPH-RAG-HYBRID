static V9fsFidState *lookup_fid(V9fsState *s, int32_t fid)

{

    V9fsFidState *f;



    for (f = s->fid_list; f; f = f->next) {

        if (f->fid == fid) {

            v9fs_do_setuid(s, f->uid);

            return f;

        }

    }



    return NULL;

}
