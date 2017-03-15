package main

import (
	"fmt"
	"math/rand"

	"github.com/davecgh/go-spew/spew"
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

func pinv(base []float64, i, j int) mat64.Dense {

	x := mat64.NewDense(i, j, base)
	var SVD mat64.SVD
	var um, vm, vw, vwu mat64.Dense

	SVD.Factorize(x, matrix.SVDFull)
	um.UFromSVD(&SVD)
	vm.VFromSVD(&SVD)
	w := SVD.Values(nil)

	wm := mat64.NewDense(i, j, nil)
	leng := len(w)
	for i, v := range w {
		for j := 0; j < leng; j++ {
			if i == j && v != 0 {
				wm.Set(i, j, 1/v)
			} else {
				wm.Set(i, j, 0.0)
			}
		}
	}
	vw.Mul(&vm, wm)
	vwu.Mul(&vw, &um)
	fa := mat64.Formatted(&vwu, mat64.Prefix("                 "), mat64.Excerpt(2))
	fmt.Printf("Pseudo Inverse :\npseudo Inverse = %v \n", fa)
	return (vwu)

}

func main() {

	elem2 := []float64{2, 4, 1, 2}
	x := mat64.NewDense(2, 2, elem2)

	rets := pinv(elem2, 2, 2)
	spew.Println(rets)

	var invck mat64.Dense
	invx1 := invck.Inverse(x)
	if invx1 != nil {
		spew.Println(invx1)
	}
	fa := mat64.Formatted(&invck, mat64.Prefix("    "), mat64.Squeeze())
	fmt.Printf("INV TEST:\na = %v\n\n", fa)

	data := make([]float64, 36)
	for i := range data {
		data[i] = rand.NormFloat64()
	}

	rets2 := pinv(data, 6, 6)
	spew.Println(rets2)

}
