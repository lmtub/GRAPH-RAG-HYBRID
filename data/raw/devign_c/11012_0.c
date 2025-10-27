static int v9fs_complete_rename(V9fsState *s, V9fsRenameState *vs)

{

    int err = 0;

    char *old_name, *new_name;

    char *end;



    if (vs->newdirfid != -1) {

        V9fsFidState *dirfidp;

        dirfidp = lookup_fid(s, vs->newdirfid);



        if (dirfidp == NULL) {

            err = -ENOENT;

            goto out;

        }



        BUG_ON(dirfidp->fid_type != P9_FID_NONE);



        new_name = qemu_mallocz(dirfidp->path.size + vs->name.size + 2);



        strcpy(new_name, dirfidp->path.data);

        strcat(new_name, "/");

        strcat(new_name + dirfidp->path.size, vs->name.data);

    } else {

        old_name = vs->fidp->path.data;

        end = strrchr(old_name, '/');

        if (end) {

            end++;

        } else {

            end = old_name;

        }

        new_name = qemu_mallocz(end - old_name + vs->name.size + 1);



        strncat(new_name, old_name, end - old_name);

        strncat(new_name + (end - old_name), vs->name.data, vs->name.size);

    }



    v9fs_string_free(&vs->name);

    vs->name.data = qemu_strdup(new_name);

    vs->name.size = strlen(new_name);



    if (strcmp(new_name, vs->fidp->path.data) != 0) {

        if (v9fs_do_rename(s, &vs->fidp->path, &vs->name)) {

            err = -errno;

        } else {

            V9fsFidState *fidp;

            /*

            * Fixup fid's pointing to the old name to

            * start pointing to the new name

            */

            for (fidp = s->fid_list; fidp; fidp = fidp->next) {

                if (vs->fidp == fidp) {

                    /*

                    * we replace name of this fid towards the end

                    * so that our below strcmp will work

                    */

                    continue;

                }

                if (!strncmp(vs->fidp->path.data, fidp->path.data,

                    strlen(vs->fidp->path.data))) {

                    /* replace the name */

                    v9fs_fix_path(&fidp->path, &vs->name,

                                  strlen(vs->fidp->path.data));

                }

            }

            v9fs_string_copy(&vs->fidp->path, &vs->name);

        }

    }

out:

    v9fs_string_free(&vs->name);

    return err;

}
