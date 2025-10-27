static int handle_renames_and_mkdirs(BDRVVVFATState* s)

{

    int i;



#ifdef DEBUG

    fprintf(stderr, "handle_renames\n");

    for (i = 0; i < s->commits.next; i++) {

	commit_t* commit = array_get(&(s->commits), i);

	fprintf(stderr, "%d, %s (%d, %d)\n", i, commit->path ? commit->path : "(null)", commit->param.rename.cluster, commit->action);

    }

#endif



    for (i = 0; i < s->commits.next;) {

	commit_t* commit = array_get(&(s->commits), i);

	if (commit->action == ACTION_RENAME) {

	    mapping_t* mapping = find_mapping_for_cluster(s,

		    commit->param.rename.cluster);

	    char* old_path = mapping->path;



	    assert(commit->path);

	    mapping->path = commit->path;

	    if (rename(old_path, mapping->path))

		return -2;



	    if (mapping->mode & MODE_DIRECTORY) {

		int l1 = strlen(mapping->path);

		int l2 = strlen(old_path);

		int diff = l1 - l2;

		direntry_t* direntry = array_get(&(s->directory),

			mapping->info.dir.first_dir_index);

		uint32_t c = mapping->begin;

		int i = 0;



		/* recurse */

		while (!fat_eof(s, c)) {

		    do {

			direntry_t* d = direntry + i;



			if (is_file(d) || (is_directory(d) && !is_dot(d))) {

			    mapping_t* m = find_mapping_for_cluster(s,

				    begin_of_direntry(d));

			    int l = strlen(m->path);

			    char* new_path = g_malloc(l + diff + 1);



			    assert(!strncmp(m->path, mapping->path, l2));



                            pstrcpy(new_path, l + diff + 1, mapping->path);

                            pstrcpy(new_path + l1, l + diff + 1 - l1,

                                    m->path + l2);



			    schedule_rename(s, m->begin, new_path);

			}

			i++;

		    } while((i % (0x10 * s->sectors_per_cluster)) != 0);

		    c = fat_get(s, c);

		}

	    }



	    free(old_path);

	    array_remove(&(s->commits), i);

	    continue;

	} else if (commit->action == ACTION_MKDIR) {

	    mapping_t* mapping;

	    int j, parent_path_len;



#ifdef __MINGW32__

            if (mkdir(commit->path))

                return -5;

#else

            if (mkdir(commit->path, 0755))

                return -5;

#endif



	    mapping = insert_mapping(s, commit->param.mkdir.cluster,

		    commit->param.mkdir.cluster + 1);

	    if (mapping == NULL)

		return -6;



	    mapping->mode = MODE_DIRECTORY;

	    mapping->read_only = 0;

	    mapping->path = commit->path;

	    j = s->directory.next;

	    assert(j);

	    insert_direntries(s, s->directory.next,

		    0x10 * s->sectors_per_cluster);

	    mapping->info.dir.first_dir_index = j;



	    parent_path_len = strlen(commit->path)

		- strlen(get_basename(commit->path)) - 1;

	    for (j = 0; j < s->mapping.next; j++) {

		mapping_t* m = array_get(&(s->mapping), j);

		if (m->first_mapping_index < 0 && m != mapping &&

			!strncmp(m->path, mapping->path, parent_path_len) &&

			strlen(m->path) == parent_path_len)

		    break;

	    }

	    assert(j < s->mapping.next);

	    mapping->info.dir.parent_mapping_index = j;



	    array_remove(&(s->commits), i);

	    continue;

	}



	i++;

    }

    return 0;

}
