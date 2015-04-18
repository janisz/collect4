package collect4
import (
	"github.com/op/go-logging"
	"os"
)

var log = logging.MustGetLogger("example")
var format = logging.MustStringFormatter(
"%{color}%{time:15:04:05.000} %{shortfunc} ▶ %{level:.4s} %{id:03x}%{color:reset} %{message}",
)

func ConfigureDebugLogging() {
	var backend logging.Backend
	// For demo purposes, create two backend for os.Stderr.
	backend = logging.NewLogBackend(os.Stderr, "", 0)

	// For messages written to backend we want to add some additional
	// information to the output, including the used log level and the name of
	// the function.
	backendFormatter := logging.NewBackendFormatter(backend, format)

	// Only errors and more severe messages should be sent to backend1
	backend1Leveled := logging.AddModuleLevel(backendFormatter)
	backend1Leveled.SetLevel(logging.DEBUG, "")

	// Set the backends to be used.
	logging.SetBackend(backend1Leveled)
}

func ConfigureInfoLogging() {
	var backend logging.Backend
	// For demo purposes, create two backend for os.Stderr.
	backend = logging.NewLogBackend(os.Stderr, "", 0)

	// For messages written to backend we want to add some additional
	// information to the output, including the used log level and the name of
	// the function.
	backendFormatter := logging.NewBackendFormatter(backend, format)

	// Only errors and more severe messages should be sent to backend1
	backend1Leveled := logging.AddModuleLevel(backendFormatter)
	backend1Leveled.SetLevel(logging.INFO, "")

	// Set the backends to be used.
	logging.SetBackend(backend1Leveled)
}
