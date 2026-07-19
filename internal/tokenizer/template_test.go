package tokenizer

import (
	"strings"
	"testing"
)

func TestRecognizeTemplate(t *testing.T) {
	if !RecognizeTemplate(qwen3GuardTemplate) {
		t.Fatal("pinned template not recognized")
	}
	reformatted := strings.ReplaceAll(qwen3GuardTemplate, "\n    ", "\n        ")
	if !RecognizeTemplate(reformatted) {
		t.Fatal("whitespace-only change rejected")
	}
	mutated := strings.Replace(qwen3GuardTemplate, "Jailbreak", "Malware", 1)
	if RecognizeTemplate(mutated) {
		t.Fatal("policy mutation accepted")
	}
	if RecognizeTemplate("chatml") {
		t.Fatal("unrelated template accepted")
	}
}

func TestFormatUserModeration(t *testing.T) {
	out, err := FormatQwen3Guard([]Message{
		{Role: RoleUser, Content: "hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := qwen3GuardUserPrefix + "USER: hello" + qwen3GuardUserSuffix
	if out != want {
		t.Fatalf("format mismatch:\n%q\nwant:\n%q", out, want)
	}
}

func TestFormatAssistantModeration(t *testing.T) {
	out, err := FormatQwen3Guard([]Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, Content: "a"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := qwen3GuardAssistantPrefix + "USER: q\n\nASSISTANT: a" + qwen3GuardAssistantSuffix
	if out != want {
		t.Fatalf("format mismatch:\n%q\nwant:\n%q", out, want)
	}
}

func TestFormatSystemThenUser(t *testing.T) {
	out, err := FormatQwen3Guard([]Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "q"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := qwen3GuardUserPrefix + "USER: sys\n\nq" + qwen3GuardUserSuffix
	if out != want {
		t.Fatalf("format mismatch:\n%q\nwant:\n%q", out, want)
	}
}

func TestFormatLongerConversation(t *testing.T) {
	out, err := FormatQwen3Guard([]Message{
		{Role: RoleSystem, Content: "sys"},
		{Role: RoleUser, Content: "u1"},
		{Role: RoleAssistant, Content: "a1"},
		{Role: RoleUser, Content: "u2"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := qwen3GuardUserPrefix + "USER: sys\n\nu1\n\nASSISTANT: a1\n\nUSER: u2" + qwen3GuardUserSuffix
	if out != want {
		t.Fatalf("format mismatch:\n%q\nwant:\n%q", out, want)
	}
}

func TestFormatValidation(t *testing.T) {
	if _, err := FormatQwen3Guard(nil); err == nil {
		t.Fatal("empty conversation accepted")
	}
	if _, err := FormatQwen3Guard([]Message{{Role: "tool", Content: "x"}}); err == nil {
		t.Fatal("unknown role accepted")
	}
	if _, err := FormatQwen3Guard([]Message{{Role: RoleSystem, Content: "sys"}}); err == nil {
		t.Fatal("system-only conversation accepted")
	}
}

func TestTokenizerFormatDispatch(t *testing.T) {
	tk, err := New(testMeta())
	if err != nil {
		t.Fatal(err)
	}
	out, err := tk.Format([]Message{{Role: RoleUser, Content: "hello"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out, qwen3GuardUserPrefix) {
		t.Fatalf("format = %q", out)
	}
}
