static void _ccv_write_binary_fd(ccv_dense_matrix_t* mat, FILE* fd, void* conf)
{
	fwrite("CCVBINDM", 1, 8, fd);
	int ctype = mat->type & 0xFFFFF;
	fwrite(&ctype, 1, 4, fd);
	fwrite(&(mat->rows), 1, 4, fd);
	fwrite(&(mat->cols), 1, 4, fd);
	fwrite(mat->data.u8, 1, mat->step * mat->rows, fd);
	fflush(fd);
}

static int _ccv_write_plain_stream(ccv_dense_matrix_t* mat, void* buf, size_t len)
{
	if (len < 20 + mat->step * mat->rows)
		return -1;
	memcpy(buf, "CCVBINDM", 8);
	int ctype = mat->type & 0xFFFFF;
	memcpy(buf + 8, &ctype, 4);
	memcpy(buf + 12, &mat->rows, 4);
	memcpy(buf + 16, &mat->cols, 4);
	memcpy(buf + 20, mat->data.u8, mat->step * mat->rows);
	return 0;
}

static void _ccv_read_binary_fd(FILE* in, ccv_dense_matrix_t** x, int type)
{
	fseek(in, 8, SEEK_SET);
	fread(&type, 1, 4, in);
	int rows, cols;
	fread(&rows, 1, 4, in);
	fread(&cols, 1, 4, in);
	*x = ccv_dense_matrix_new(rows, cols, type, 0, 0);
	fread((*x)->data.u8, 1, (*x)->step * (*x)->rows, in);
}
