package main

import (
	"fmt"

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

	wm := mat64.NewDense(2, 2, nil)
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
	// elem := []float64{2, 5, 1, 3}
	// x := mat64.NewDense(2, 2, elem)

	elem2 := []float64{2, 4, 1, 2}
	x2 := mat64.NewDense(2, 2, elem2)

	rets := pinv(elem2, 2, 2)
	spew.Println(rets)

	var invck mat64.Dense
	invx1 := invck.Inverse(x2)
	if invx1 != nil {
		spew.Println(invx1)
	}
	fa := mat64.Formatted(&invck, mat64.Prefix("    "), mat64.Squeeze())
	fmt.Printf("INV TEST:\na = %v\n\n", fa)

	var m3 mat64.SVD
	m3.Factorize(x2, matrix.SVDFull)
	var um2, vm2, s2m, s3m, inv, inv2 mat64.Dense
	um2.UFromSVD(&m3)
	vm2.VFromSVD(&m3)
	inv.Inverse(&um2)
	s2m.Mul(&inv, x2)
	s3m.Mul(&s2m, &vm2)
	inv2.Inverse(&s3m)

	m4 := mat64.NewDense(2, 2, nil)

}
