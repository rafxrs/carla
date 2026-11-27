package osp.jsettlers;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * An Open Seat Agent run as a process: messages go to its standard input, one JSON object per line, its answers come
 * from its standard output; lines there that aren't JSON objects are skipped, standard error goes to a log file if given.
 */
public final class AgentProcess implements AutoCloseable
{
    private final Process process;
    private final BufferedWriter input;
    private final LinkedBlockingQueue<Object> lines = new LinkedBlockingQueue<>();
    private static final Object END = new Object();

    public AgentProcess(String commandLine, File errorLog) throws IOException
    {
        ProcessBuilder builder = new ProcessBuilder(split(commandLine));
        if (errorLog != null)
            builder.redirectError(ProcessBuilder.Redirect.appendTo(errorLog));
        else
            builder.redirectError(ProcessBuilder.Redirect.DISCARD);
        process = builder.start();
        input = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
        Thread reader = new Thread(() ->
        {
            try (BufferedReader output = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8)))
            {
                String line;
                while ((line = output.readLine()) != null)
                    lines.add(line);
            }
            catch (IOException e) {}
            lines.add(END);
        }, "agent-output");
        reader.setDaemon(true);
        reader.start();
    }

    /** The words of a command line: separated by spaces, double quotes group words with spaces */
    public static List<String> split(String commandLine)
    {
        List<String> words = new ArrayList<>();
        StringBuilder word = new StringBuilder();
        boolean quoted = false, any = false;
        for (char c : commandLine.toCharArray())
        {
            if (c == '"') { quoted = ! quoted; any = true; }
            else if (Character.isWhitespace(c) && ! quoted)
            {
                if (any)
                    words.add(word.toString());
                word.setLength(0);
                any = false;
            }
            else { word.append(c); any = true; }
        }
        if (any)
            words.add(word.toString());
        return words;
    }

    public void send(Map<String, Object> message) throws IOException
    {
        input.write(Json.write(message));
        input.write('\n');
        input.flush();
    }

    /** No answer within the time */
    public static final class Timeout extends IOException
    {
        Timeout(String message)
        {
            super(message);
        }
    }

    /** The agent's next answer within the timeout (else Timeout); an ended agent is an IOException */
    @SuppressWarnings("unchecked")
    public Map<String, Object> receive(long timeoutMillis) throws IOException, InterruptedException
    {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (true)
        {
            Object line = lines.poll(Math.max(1, deadline - System.currentTimeMillis()), TimeUnit.MILLISECONDS);
            if (line == null)
                throw new Timeout("The Open Seat Agent gave no answer within " + timeoutMillis + " ms");
            if (line == END)
            {
                lines.add(END);
                throw new IOException("The Open Seat Agent ended");
            }
            String text = ((String) line).trim();
            if (text.isEmpty())
                continue;
            try
            {
                Object value = Json.read(text);
                if (value instanceof Map)
                    return (Map<String, Object>) value;
            }
            catch (IllegalArgumentException e) {}
        }
    }

    @Override
    public void close()
    {
        try { input.close(); } catch (IOException e) {}
        try
        {
            if (! process.waitFor(5, TimeUnit.SECONDS))
                process.destroyForcibly().waitFor(5, TimeUnit.SECONDS);
        }
        catch (InterruptedException e)
        {
            process.destroyForcibly();
        }
    }
}
