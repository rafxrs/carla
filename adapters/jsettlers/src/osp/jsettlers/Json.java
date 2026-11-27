package osp.jsettlers;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The little JSON the Open Seat Protocol needs: objects as maps, arrays as lists, integers as Long
 * (Integer and Long written alike), strings, booleans and null; written compactly, read from one
 * line.
 */
public final class Json {
  private Json() {}

  /**
   * The JSON text of a value; maps keep their order, unless sorted is set (for comparing values as
   * text)
   */
  public static String write(Object value, boolean sorted) {
    StringBuilder out = new StringBuilder();
    write(value, sorted, out);
    return out.toString();
  }

  public static String write(Object value) {
    return write(value, false);
  }

  @SuppressWarnings("unchecked")
  private static void write(Object value, boolean sorted, StringBuilder out) {
    if (value == null) out.append("null");
    else if (value instanceof Boolean || value instanceof Integer || value instanceof Long)
      out.append(value);
    else if (value instanceof Float || value instanceof Double) {
      if (!Double.isFinite(((Number) value).doubleValue()))
        throw new IllegalArgumentException("Non-finite JSON number");
      out.append(value);
    } else if (value instanceof String) string((String) value, out);
    else if (value instanceof Map) {
      Map<String, Object> map = (Map<String, Object>) value;
      if (sorted) map = new TreeMap<>(map);
      out.append('{');
      boolean first = true;
      for (Map.Entry<String, Object> entry : map.entrySet()) {
        if (!first) out.append(',');
        first = false;
        string(entry.getKey(), out);
        out.append(':');
        write(entry.getValue(), sorted, out);
      }
      out.append('}');
    } else if (value instanceof List) {
      out.append('[');
      Iterator<Object> items = ((List<Object>) value).iterator();
      while (items.hasNext()) {
        write(items.next(), sorted, out);
        if (items.hasNext()) out.append(',');
      }
      out.append(']');
    } else if (value instanceof int[]) {
      out.append('[');
      int[] items = (int[]) value;
      for (int i = 0; i < items.length; i++) out.append(i == 0 ? "" : ",").append(items[i]);
      out.append(']');
    } else throw new IllegalArgumentException("No JSON for " + value.getClass());
  }

  private static void string(String text, StringBuilder out) {
    out.append('"');
    for (char c : text.toCharArray()) {
      switch (c) {
        case '"':
          out.append("\\\"");
          break;
        case '\\':
          out.append("\\\\");
          break;
        case '\n':
          out.append("\\n");
          break;
        case '\r':
          out.append("\\r");
          break;
        case '\t':
          out.append("\\t");
          break;
        default:
          if (c < 0x20) out.append(String.format("\\u%04x", (int) c));
          else out.append(c);
      }
    }
    out.append('"');
  }

  /** The value of one JSON text */
  public static Object read(String text) {
    Reader reader = new Reader(text);
    Object value = reader.value();
    reader.space();
    if (reader.at < text.length())
      throw new IllegalArgumentException("Text after the JSON value at " + reader.at);
    return value;
  }

  /** Builds an object from key, value, key, value, ... in that order */
  public static Map<String, Object> object(Object... keysAndValues) {
    Map<String, Object> map = new LinkedHashMap<>();
    for (int i = 0; i < keysAndValues.length; i += 2)
      map.put((String) keysAndValues[i], keysAndValues[i + 1]);
    return map;
  }

  /** Builds an array of the values */
  public static List<Object> array(Object... values) {
    List<Object> list = new ArrayList<>();
    for (Object value : values) list.add(value);
    return list;
  }

  /** A JSON number as an int */
  public static int integer(Object number) {
    return ((Number) number).intValue();
  }

  private static final class Reader {
    private final String text;
    private int at;

    Reader(String text) {
      this.text = text;
    }

    void space() {
      while (at < text.length() && Character.isWhitespace(text.charAt(at))) at++;
    }

    Object value() {
      space();
      if (at >= text.length()) throw new IllegalArgumentException("JSON ends early");
      char c = text.charAt(at);
      switch (c) {
        case '{':
          return object();
        case '[':
          return array();
        case '"':
          return string();
        case 't':
          expect("true");
          return Boolean.TRUE;
        case 'f':
          expect("false");
          return Boolean.FALSE;
        case 'n':
          expect("null");
          return null;
        default:
          return number();
      }
    }

    private void expect(String word) {
      if (!text.startsWith(word, at))
        throw new IllegalArgumentException("Expected " + word + " at " + at);
      at += word.length();
    }

    private Map<String, Object> object() {
      Map<String, Object> map = new LinkedHashMap<>();
      at++;
      space();
      if (text.charAt(at) == '}') {
        at++;
        return map;
      }
      while (true) {
        space();
        String key = string();
        space();
        expect(":");
        map.put(key, value());
        space();
        char c = text.charAt(at++);
        if (c == '}') return map;
        if (c != ',') throw new IllegalArgumentException("Expected , or } at " + (at - 1));
      }
    }

    private List<Object> array() {
      List<Object> list = new ArrayList<>();
      at++;
      space();
      if (text.charAt(at) == ']') {
        at++;
        return list;
      }
      while (true) {
        list.add(value());
        space();
        char c = text.charAt(at++);
        if (c == ']') return list;
        if (c != ',') throw new IllegalArgumentException("Expected , or ] at " + (at - 1));
      }
    }

    private String string() {
      if (text.charAt(at) != '"') throw new IllegalArgumentException("Expected a string at " + at);
      at++;
      StringBuilder out = new StringBuilder();
      while (true) {
        char c = text.charAt(at++);
        if (c == '"') return out.toString();
        if (c != '\\') {
          out.append(c);
          continue;
        }
        char e = text.charAt(at++);
        switch (e) {
          case 'n':
            out.append('\n');
            break;
          case 'r':
            out.append('\r');
            break;
          case 't':
            out.append('\t');
            break;
          case 'b':
            out.append('\b');
            break;
          case 'f':
            out.append('\f');
            break;
          case 'u':
            out.append((char) Integer.parseInt(text.substring(at, at + 4), 16));
            at += 4;
            break;
          default:
            out.append(e);
        }
      }
    }

    private Object number() {
      int start = at;
      while (at < text.length() && "+-0123456789.eE".indexOf(text.charAt(at)) >= 0) at++;
      String number = text.substring(start, at);
      if (number.isEmpty()) throw new IllegalArgumentException("Unexpected character at " + start);
      if (number.contains(".") || number.contains("e") || number.contains("E"))
        return (long) Double.parseDouble(number);
      return Long.parseLong(number);
    }
  }
}
