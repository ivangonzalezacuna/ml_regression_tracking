package main

import (
	"log"
	"ml/logisticregression"
)

func main() {
	if err := logisticregression.Run(); err != nil {
		log.Fatal(err)
	}
}
