//go:build !unix

package backing

func mapFile(string) ([]byte, func([]byte) error, error) {
	return nil, nil, errMappingUnsupported
}
