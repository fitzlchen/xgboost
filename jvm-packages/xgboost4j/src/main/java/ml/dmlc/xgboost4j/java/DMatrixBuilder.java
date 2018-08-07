package ml.dmlc.xgboost4j.java;

import ml.dmlc.xgboost4j.LabeledPoint;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.concurrent.CompletableFuture;

/**
 * Created by: fitz
 * <p>
 * Date: 2018/3/4
 * <p>
 * Description:
 */
public class DMatrixBuilder {
    private LinkedList<Long> headerList; private LinkedList<Integer> indexList; private LinkedList<Float> dataList;
    private int nrow = 0, ncol= 0;
    private MATRIX_FORMAT format;

    public LinkedList<Long> getHeaderList() {
        return headerList;
    }

    public LinkedList<Integer> getIndexList() {
        return indexList;
    }

    public LinkedList<Float> getDataList() {
        return dataList;
    }

    public DMatrixBuilder(MATRIX_FORMAT format){
        this.format = format;
        headerList = new LinkedList<>();
        indexList = new LinkedList<>();
        dataList = new LinkedList<>();
    }

    public MATRIX_FORMAT getFormat(){
        return format;
    }

    /**
     * invoke while reading per record
     * fill the indices array with the index of elements.
     * When all the elements in a record are zero, then fill an empty int array or null.
     *
     * @param indices
     * @return
     */
    public DMatrixBuilder addIndices(int[] indices){
        if (indexList == null)
            indexList = new LinkedList<>();
        if (indices == null || indices.length == 0){
            addHearders(0);
            return this;
        }
        for (int i : indices)
            indexList.add(i);
        addHearders(indices.length);
        return this;
    }

    public void reset(){
        headerList.clear();
        indexList.clear();
        dataList.clear();
        ncol = 0;
        nrow = 0;
    }

    private void addHearders(long size){
        if (headerList.size() == 0){
            headerList.add(0L);
        }
        headerList.add(headerList.getLast() + size);
    }

    /**
     *
     * @param data
     * @return
     */
    public DMatrixBuilder addDatas(float[] data){
        if (dataList == null)
            dataList = new LinkedList<>();
        if (data == null || data.length == 0)
            return null;
        for (float i : data)
            dataList.add(i);
        return this;
    }

    /**
     * create dense DMatrix or sparse DMatrix
     *
     * @return DMatrix
     * @throws XGBoostError,DMatrixSizeMismatchException,UnknownMatrixTypeException
     */
    public DMatrix build() throws XGBoostError, DMatrixSizeMismatchException, UnknownMatrixTypeException {
        DMatrix mat = null;
        long[] headers; int[] indices; float[] data;
        switch (format){
            case CSR:
                if (indexList.size() != dataList.size())
                    throw new DMatrixSizeMismatchException("Indices and data are not the same size!");
                headers = new long[headerList.size()];
                indices = new int[indexList.size()];
                data = new float[dataList.size()];

                for (int i = 0; i < headers.length; i++)
                    headers[i] = headerList.get(i);
                for (int i = 0; i < indices.length; i++){
                    indices[i] = indexList.get(i);
                    data[i] = dataList.get(i);
                }

                mat = new DMatrix(headers, indices, data, DMatrix.SparseType.CSR, 0);
                break;
            case CSC:
                if (indexList.size() != dataList.size())
                    throw new DMatrixSizeMismatchException("Indices and data are not the same size!");
                headers = new long[headerList.size()];
                indices = new int[indexList.size()];
                data = new float[dataList.size()];

                for (int i = 0; i < headers.length; i++)
                    headers[i] = headerList.get(i);
                for (int i = 0; i < indices.length; i++){
                    indices[i] = indexList.get(i);
                    data[i] = dataList.get(i);
                }

                mat = new DMatrix(headers, indices, data, DMatrix.SparseType.CSC, 0);
                break;
            case DENSE:
                data = new float[dataList.size()];

                for (int i = 0; i < dataList.size(); i++)
                    data[i] = dataList.get(i);

                mat = new DMatrix(data, nrow, ncol);
                break;
            case NONE:
                throw new UnknownMatrixTypeException("The format of DMatrix is required!");
        }

        return mat;
    }

    /**
     *
     *
     * @param iter
     * @param cacheInfo
     * @return
     * @throws XGBoostError
     */
    public DMatrix createFromIterator(Iterator<LabeledPoint> iter, String cacheInfo) throws XGBoostError {
        return new DMatrix(iter, cacheInfo);
    }

    /**
     * Create DMatrix by loading libsvm file from dataPath
     *
     * @param dataPath The path to the data.
     * @throws XGBoostError
     */
    public DMatrix createFromLibsvm(String dataPath) throws XGBoostError {
        return new DMatrix(dataPath);
    }

    /**
     *
     * @param data
     * @return
     */
    //TODO
    public DMatrixBuilder addDenseData(float[] data){
        if (ncol == 0)
            ncol = data.length;
        nrow++;

        return this;
    }

    /**
     *
     * @param val
     * @return
     */
    public DMatrixBuilder fillMissing(float val){
        return this;
    }

    private class Point{
        private long x,y;
        private float val;

        Point(long x, long y, float val){
            this.x = x;
            this.y = y;
            this.val = val;
        }
    }

    public enum MATRIX_FORMAT{
        CSR,CSC,DENSE,NONE
    }

    static class DMatrixParam{
        private int[] indices;
        private float[] data;
        private CompletableFuture<float[]> future;

        public DMatrixParam(int[] indices, float[] data, CompletableFuture<float[]> future) {
            this.indices = indices;
            this.data = data;
            this.future = future;
        }

        public int[] getIndices() {
            return indices;
        }

        public void setIndices(int[] indices) {
            this.indices = indices;
        }

        public float[] getData() {
            return data;
        }

        public void setData(float[] data) {
            this.data = data;
        }

        public CompletableFuture<float[]> getFuture() {
            return future;
        }

        public void setFuture(CompletableFuture<float[]> future) {
            this.future = future;
        }
    }
}
